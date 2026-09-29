"""数据集体检：给"刚生成的一套 npz"做结构与一致性验收，逐项 Pass/Fail。

用法
----
本机：
    E:\\Anaconda\\envs\\dsca\\python.exe verify_dataset.py ^
        --data E:\\Datasets\\DSCA\\processed960 --expect-size 960

集群（解包后第一件事就跑这个）：
    python verify_dataset.py --data $HOME/dsca_data/processed960 --expect-size 960

为什么需要它
------------
`prepare.py` 跑完只保证"脚本没崩"，但从生成到进训练中间还有三个**静默**失效：

  1. **单例失败被吞掉**。`prepare.py` 对每例是 try/except 的：某例抛异常（DICOM 读不动、
     尺寸异常…）只会记进 report.json 然后**继续跑**。所以"跑完了"≠"180 例都成功了"。
  2. **tar / SFTP / 解包 断在中途**。大文件最容易在这里少几个 —— 一个截断的 npz
     要么 `np.load` 报错，要么读出形状不对。
  3. **传错版本**。processed(512) 当 processed768 传，训练照样跑得起来、loss 照样降、
     Dice 照样像模像样 —— 只是全部对不上号。`--expect-size` 就是拦这个的。

第 3 条已经被咬过一次（4 折/5 折那次就是 glob 写漏），所以宁可跑 30 秒体检。

检查项
------
  [1] 结构      train/ test/ folds.json 齐不齐
  [2] 文件数    train=180 / test=44
  [3] 折划分    folds.json 的 sha256 与 512/768 版**逐字节一致**
                （不一致 = 折划分变了 ⇒ 不能和已有 run 做同折 A/B）
  [4] 逐例结构  seq (8,N,N) uint16 / minip (N,N) uint16 / label (N,N) uint8
  [5] 类别合法  label 取值 ⊆ {0,1,2}（冒出 3 说明最近邻插值把类别串了）
  [6] 对齐 AUC  抽样算 label vs 该样本 minip 的 AUC，必须远大于 0.5
                —— 这是**轴序 bug 的兜底探测器**（错位 90° 时 AUC ≈ 0.5）
  [7] 前景占比  BV/MAT 像素占比要落在合理带内
  [8] 缺类名单  test 里"GT 无 MAT"的必须**恰好**是那 8 例
                —— 集合不对说明样本被重排/替换过，macro 口径的分母就变了

退出码：0 = 全过；1 = 有 FAIL。CI / 流水线里可直接判。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.prepare import _align_auc                          # noqa: E402

# 512 / 768 / 960 三套数据的 folds.json 必须**逐字节相同**（都是 --folds 5 --seed 0
# 在同一个 core 列表上算出来的）。变了就说明折划分被改动过。
EXPECTED_FOLDS_SHA256 = "846f4ebe54a5b6e7b9f3545efb10a02ced44e2147712593ca8cca6e28cc96279"

# test 里"GT 完全没有 MAT（class 2）"的样本，论文的 macro 口径分母 = 44 - 8 = 36。
EXPECTED_NO_MAT_TEST = {"GJW_AP_05", "GJW_AP_35", "GJW_L_14", "GJW_L_41",
                        "LXJ_AP_07", "LXJ_AP_32", "LXJ_L_51", "LYL_AL_20"}

N_TRAIN, N_TEST, N_FRAMES = 180, 44, 8


class Report:
    """攒结果 + 立刻回显，最后统一打表。"""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<26} {detail}", flush=True)
        return bool(ok)

    def finish(self) -> int:
        bad = [r for r in self.rows if not r[1]]
        print()
        print("=" * 78)
        if bad:
            print(f"体检结果：❌ {len(bad)}/{len(self.rows)} 项未通过")
            for name, _, detail in bad:
                print(f"   - {name}: {detail}")
            print("=" * 78)
            return 1
        print(f"体检结果：✅ {len(self.rows)}/{len(self.rows)} 项全部通过")
        print("=" * 78)
        return 0


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="DSCA 数据集体检")
    ap.add_argument("--data", required=True, help="数据集根目录（含 train/ test/ folds.json）")
    ap.add_argument("--expect-size", type=int, default=0,
                    help="期望的边长；0 = 不校验（不安全）")
    ap.add_argument("--auc-n", type=int, default=10, help="抽样算对齐 AUC 的例数")
    ap.add_argument("--auc-min", type=float, default=0.85,
                    help="抽样里最差的 AUC 允许低到多少（错位 90° 时≈0.5）")
    ap.add_argument("--folds-sha", default=EXPECTED_FOLDS_SHA256)
    ap.add_argument("--skip-sha", action="store_true", help="不做折划分 sha 校验")
    args = ap.parse_args()

    root = Path(args.data)
    rep = Report()
    print("=" * 78)
    print(f"  数据集体检  {root}")
    print(f"  期望边长    {args.expect_size or '（未指定）'}")
    print("=" * 78)

    # ---- [1] 结构 ----------------------------------------------------------
    train_dir, test_dir = root / "train", root / "test"
    folds_path = root / "folds.json"
    ok_struct = train_dir.is_dir() and test_dir.is_dir() and folds_path.is_file()
    rep.add("1 结构", ok_struct,
            f"train={train_dir.is_dir()} test={test_dir.is_dir()} folds={folds_path.is_file()}")
    if not ok_struct:
        return rep.finish()

    train_npz = sorted(train_dir.glob("*.npz"))
    test_npz = sorted(test_dir.glob("*.npz"))

    # ---- [2] 文件数 --------------------------------------------------------
    rep.add("2 文件数", len(train_npz) == N_TRAIN and len(test_npz) == N_TEST,
            f"train {len(train_npz)}/{N_TRAIN}  test {len(test_npz)}/{N_TEST}")

    # ---- [3] 折划分 --------------------------------------------------------
    if not args.skip_sha:
        sha = sha256_of(folds_path)
        rep.add("3 折划分 sha256", sha == args.folds_sha,
                f"{sha[:16]}…" + ("" if sha == args.folds_sha else f"  期望 {args.folds_sha[:16]}…"))

    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    fold_n = [len(v["val"]) for v in folds.values()]
    val_cores = {c for v in folds.values() for c in v["val"]}
    all_cores = {c for v in folds.values() for c in v["train"]} | val_cores
    name_ok = {p.stem for p in train_npz} == all_cores
    rep.add("3b 折覆盖全部训练样本", name_ok,
            f"folds {len(all_cores)} 例 / npz {len(train_npz)} 例；各折 val {fold_n} 极差 {max(fold_n)-min(fold_n)}")

    # ---- [4][5] 逐例结构 + 类别 --------------------------------------------
    bad_shape: list[str] = []
    bad_class: list[str] = []
    sizes_seen: set[int] = set()
    n_fg = n_bv = n_mat = n_tot = 0
    no_mat: list[str] = []

    for p in test_npz + train_npz:
        try:
            z = np.load(p)
            seq, minip, label = z["seq"], z["minip"], z["label"]
        except Exception as e:                                       # noqa: BLE001
            bad_shape.append(f"{p.name}: 读不动({type(e).__name__})")
            continue
        n = label.shape[0] if label.ndim == 2 else -1
        sizes_seen.add(n)
        want = (args.expect_size,) if args.expect_size else (n,)
        if not (label.ndim == 2 and minip.ndim == 2
                and seq.ndim == 3 and seq.shape[0] == N_FRAMES
                and label.shape == minip.shape
                and minip.shape == (n, n) and n in want
                and seq.dtype == np.uint16 and minip.dtype == np.uint16
                and label.dtype == np.uint8):
            bad_shape.append(f"{p.name}: seq{seq.shape}{seq.dtype} minip{minip.shape} label{label.shape}{label.dtype}")
            continue
        mx = int(label.max())
        if mx > 2:
            bad_class.append(f"{p.name}: max={mx}")
        if p.parent == test_dir and int((label == 2).sum()) == 0:
            no_mat.append(p.stem)
        n_fg += int((label > 0).sum()); n_bv += int((label == 1).sum())
        n_mat += int((label == 2).sum()); n_tot += label.size

    rep.add("4 逐例结构", not bad_shape,
            "全部合规" if not bad_shape else f"{len(bad_shape)} 例异常：{bad_shape[:3]}")
    if args.expect_size:
        rep.add("4b 边长一致", sizes_seen <= {args.expect_size},
                f"实测边长 {sorted(sizes_seen)}")
    rep.add("5 类别合法 (⊆0,1,2)", not bad_class,
            "全部合规" if not bad_class else f"{len(bad_class)} 例异常：{bad_class[:3]}")

    # ---- [7] 前景占比 ------------------------------------------------------
    fg = 100.0 * n_fg / max(1, n_tot)
    bv = 100.0 * n_bv / max(1, n_tot)
    ma = 100.0 * n_mat / max(1, n_tot)
    # 512/768 实测：前景 4.78%  BV 3.76%  MAT 1.01%（三套数据几乎一致，标签是同一批渲染出来的）
    ok_fg = 4.3 <= fg <= 5.3
    rep.add("7 前景占比", ok_fg, f"前景 {fg:.3f}%（BV {bv:.3f}% + MAT {ma:.3f}%），合理带 4.3~5.3%")

    # ---- [8] 缺类名单 ------------------------------------------------------
    got_no_mat = set(no_mat)
    rep.add("8 无 MAT 的测试样本", got_no_mat == EXPECTED_NO_MAT_TEST,
            f"{len(got_no_mat)}/8 例" + ("" if got_no_mat == EXPECTED_NO_MAT_TEST
                                        else f"  多出 {sorted(got_no_mat - EXPECTED_NO_MAT_TEST)}"
                                             f"  缺少 {sorted(EXPECTED_NO_MAT_TEST - got_no_mat)}"))
    print(f"        （macro 口径的 MAT 分母 = {len(test_npz)} - {len(got_no_mat)} = {len(test_npz)-len(got_no_mat)}）")

    # ---- [6] 对齐 AUC（抽样，放最后：慢）-----------------------------------
    step = max(1, len(test_npz) // max(1, args.auc_n))
    sample = test_npz[::step][: args.auc_n]
    aucs: list[tuple[str, float]] = []
    for p in sample:
        z = np.load(p)
        a = _align_auc(z["label"], z["minip"])
        if a == a:
            aucs.append((p.stem, a))
    if aucs:
        worst = min(aucs, key=lambda kv: kv[1])
        mean = sum(a for _, a in aucs) / len(aucs)
        rep.add("6 标签-图像对齐 AUC", worst[1] >= args.auc_min,
                f"抽样 {len(aucs)} 例：均值 {mean:.4f} / 最小 {worst[1]:.4f}（{worst[0]}），下限 {args.auc_min}")
    else:
        rep.add("6 标签-图像对齐 AUC", False, "抽样全部算不出 AUC（前景太少？）")

    return rep.finish()


if __name__ == "__main__":
    sys.exit(main())
