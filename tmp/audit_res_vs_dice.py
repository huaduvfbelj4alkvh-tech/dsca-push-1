# -*- coding: utf-8 -*-
"""疑点 A 复检：原生分辨率被压到 512 是否**伤 BV 多于伤 MAT**（只读）。

为什么要重做
--------------------------------------------------------------
2026-09-26 之前的版本（旧输出见 `audit_res_vs_dice.txt.pre_axfix`）跑在**轴序 bug 期**的
eval 上（当时前景均值 0.1587），算出来的 `r = -0.19`「分辨率无关」**已被作废**。
现在 5 折干净结果（label 口径两类均 0.8549）到手，且出现了新线索：

    我们 BV 81.73  vs 论文 MinIP-only 86.45 → **-4.72**
    我们 MAT 89.26 vs 论文 MinIP-only 86.22 → **+3.04**（反而超了）

差额几乎全在 BV（小血管），MAT（主干）反而更强 —— 这**不像容量问题**，
更像"小结构在预处理里被磨掉了"。而论文的协议与我们不同（原文 IV-A）：

    "Each batch comprised 2 samples at 512 x 512 pixels which were
     **randomly cropped from images of varying dimensions**."
    "For testing ... **cropped patches of 512 x 512 pixels from the test image
     via sliding windows** ... **reconstructed to match the original image
     size**. Finally, we compared the reconstructed predictions with the GT
     **across the entire image**."

即：论文**不缩放**，训练从原生分辨率裁 512 块、测试滑窗回原生尺寸再比；
而我们 `fit_to_square()` 把长边压到 512（实测 train 69.4%、**test 81.8%** 的样本被压 1.65~2.80 倍）。

本脚本检验的**可证伪预测**
--------------------------------------------------------------
若假说成立，则跨样本应出现**剂量-反应**关系：
    原生长边越大 -> 压缩倍数越高 -> **BV Dice 下降更多，MAT 下降少得多**
判定：`r(factor, BV)` 明显比 `r(factor, MAT)` 更负，且 BV 的分组均值单调下滑。

⚠️ 这是**筛查**不是判决：n=44，其中 8 例未压缩。若趋势不显著 -> 假说不成立，
   别去改预处理的缩放 —— 那会把一个已验收通过的数据管线推倒重来。

用法
--------------------------------------------------------------
    # 集群（推荐：拿 5 折平均，比单折稳）
    python tmp/audit_res_vs_dice.py --data ~/dsca_data/processed \
        --evals 'runs/f*_axfix/eval_test.json' --out tmp/audit_res_vs_dice_axfix.txt

    # 只看一折
    python tmp/audit_res_vs_dice.py --data ~/dsca_data/processed \
        --evals runs/f0_miniponly/eval_test.json
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
ROOT = THIS.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

L: list[str] = []


def log(s: str = "") -> None:
    L.append(str(s))


# ----------------------------------------------------------------------
def load_sizes(data_root: Path) -> dict[str, tuple[int, int]]:
    """core -> (rows, cols)。优先用 processed/report.json（集群上只有它）。"""
    rp = data_root / "report.json"
    sizes: dict[str, tuple[int, int]] = {}
    if rp.exists():
        rep = json.loads(rp.read_text(encoding="utf-8"))
        for split, blk in rep.get("splits", {}).items():
            for s in blk.get("samples", []):
                core, orig = s.get("core"), s.get("orig")
                if core and orig:
                    w, h = orig.split("x")
                    sizes[core] = (int(h), int(w))     # (rows, cols)
        log(f"尺寸来源: {rp}  共 {len(sizes)} 例")
    if not sizes:
        cand = Path(r"E:\Datasets\DSCA\DSA_public\imagesTs")
        if cand.is_dir():
            try:
                import pydicom
            except ImportError:
                log("[WARN] 没有 report.json，也没有 pydicom，无法拿原始尺寸")
            else:
                for p in sorted(cand.glob("*.dcm")):
                    ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                    core = p.name.split("_", 1)[1].rsplit(".", 1)[0]
                    sizes[core] = (int(ds.Rows), int(ds.Columns))
                log(f"尺寸来源: {cand}  共 {len(sizes)} 例")
    return sizes


def gt_present(data_root: Path, split: str, core: str) -> np.ndarray | None:
    sub = "test" if split == "test" else "train"
    p = data_root / sub / f"{core}.npz"
    if not p.exists():
        return None
    lab = np.load(p)["label"]
    return np.array([(lab == c).sum() > 0 for c in (0, 1, 2)])   # 背景/BV/MAT


def mean(vals) -> float:
    v = [float(x) for x in vals if x is not None and x == x]
    return sum(v) / len(v) if v else float("nan")


def rho_perm(x: np.ndarray, y: np.ndarray, n_perm: int = 4000,
             seed: int = 0) -> tuple[float, float]:
    """Pearson r + 置换检验的 p（双尾）。n 小的时候比查表可靠。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 4 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        rp = float(np.corrcoef(x, rng.permutation(y))[0, 1])
        if abs(rp) >= abs(r):
            cnt += 1
    return r, (cnt + 1) / (n_perm + 1)


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="原生分辨率压缩 vs 各分类 Dice（只读审计）")
    ap.add_argument("--data", default=r"E:\Datasets\DSCA\processed",
                    help="processed 根目录（集群上显式传 ~/dsca_data/processed）")
    ap.add_argument("--evals", nargs="+", default=["runs/f*_axfix/eval_test.json"],
                    help="一个或多个 eval json 的 glob；给多个时对 per-core Dice 取平均")
    ap.add_argument("--out", default=str(THIS / "audit_res_vs_dice.txt"))
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    data_root = Path(os.path.expanduser(args.data))
    if not data_root.exists():
        print(f"[ERROR] --data 不存在: {data_root}")
        return 2

    files: list[str] = []
    for g in args.evals:
        files.extend(sorted(globmod.glob(g)))
    files = list(dict.fromkeys(files))
    if not files:
        print(f"[ERROR] 没匹配到 eval json: {args.evals}")
        return 2

    log("=" * 80)
    log("疑点 A 复检：原生分辨率被压到 512 是否伤 BV 多于伤 MAT")
    log("=" * 80)
    sizes = load_sizes(data_root)
    log("")

    # ---- 逐 json 收集 per-core 的 BV / MAT（label 口径：GT 没有该类就不计） ----
    per_json: list[dict] = []
    split = None
    for f in files:
        ev = json.loads(Path(f).read_text(encoding="utf-8"))
        split = ev.get("split", "test")
        n_gt_absent_mat = 0
        d: dict[str, dict[str, float]] = {}
        for r in ev.get("per_sample", []):
            core = r.get("core")
            pr = gt_present(data_root, split, core) if core else None
            bv = r.get("BV")
            mat = r.get("MAT")
            if pr is not None and not pr[2]:
                mat = None                       # GT 无 MAT -> label 口径剔除
                n_gt_absent_mat += 1
            if pr is not None and not pr[1]:
                bv = None
            d[core] = {"BV": bv, "MAT": mat}
        per_json.append(d)
        _fg = (ev.get("overall") or {}).get("mean_foreground", float("nan"))
        log(f"读入 {f}  epoch={ev.get('epoch')} tta={ev.get('tta')} "
            f"两类均(union)={_fg:.4f} "
            f"GT缺MAT {n_gt_absent_mat} 例")
    log("")

    # ---- 跨 json 平均（同一 core 的多个折的值取平均） ----
    cores = sorted({c for d in per_json for c in d})
    rows = []
    for c in cores:
        bvs = [d[c]["BV"] for d in per_json if c in d]
        mats = [d[c]["MAT"] for d in per_json if c in d]
        rows.append((c, mean(bvs), mean(mats)))

    miss = [r for r in rows if r[0] not in sizes]
    log(f"参与样本 {len(rows)} 例；尺寸缺失 {len(miss)} 例"
        + (f"（{', '.join(r[0] for r in miss[:5])}…）" if miss else ""))
    log("")

    # ---- 压缩倍数 ----
    rec = []
    for c, bv, mat in rows:
        if c not in sizes:
            continue
        rs, cs = sizes[c]
        long_side = max(rs, cs)
        factor = long_side / 512.0
        rec.append({"core": c, "long": long_side, "factor": factor,
                    "BV": bv, "MAT": mat})
    rec.sort(key=lambda r: -r["factor"])

    log("逐例（按压缩倍数降序）：")
    log("  %-14s %6s %8s %9s %9s" % ("core", "长边", "压缩倍数", "BV", "MAT"))
    for r in rec:
        log("  %-14s %6d %8.2fx %9s %9s"
            % (r["core"], r["long"], r["factor"],
               ("%.4f" % r["BV"]) if r["BV"] == r["BV"] else "  --  ",
               ("%.4f" % r["MAT"]) if r["MAT"] == r["MAT"] else "  --  "))
    log("")

    # ---- 分组（按压缩倍数） ----
    buckets = [("未被压缩 (=1.00x)", lambda f: f <= 1.0),
               ("轻度 1.0~1.9x", lambda f: 1.0 < f <= 1.9),
               ("重度 >=1.9x", lambda f: f > 1.9)]
    log("按压缩倍数分组：")
    log("  %-20s %5s %10s %10s %12s" % ("组", "例数", "BV均值", "MAT均值", "BV-MAT"))
    for name, pred in buckets:
        g = [r for r in rec if pred(r["factor"])]
        if not g:
            continue
        bv, mat = mean([r["BV"] for r in g]), mean([r["MAT"] for r in g])
        log("  %-20s %5d %10.4f %10.4f %12.4f" % (name, len(g), bv, mat, bv - mat))
    log("")

    # ---- 相关 + 置换检验 ----
    X = np.array([r["factor"] for r in rec], float)
    bv = np.array([r["BV"] for r in rec], float)
    mat = np.array([r["MAT"] for r in rec], float)
    r_bv, p_bv = rho_perm(X, bv)
    r_mat, p_mat = rho_perm(X, mat)
    log("压缩倍数 vs Dice（Pearson r + 置换检验双尾 p）：")
    log("  BV  : r = %+.3f   p = %.4f   n = %d" % (r_bv, p_bv, np.isfinite(bv).sum()))
    log("  MAT : r = %+.3f   p = %.4f   n = %d" % (r_mat, p_mat, np.isfinite(mat).sum()))
    log("")

    # ---- 判定 ----
    log("-" * 80)
    log("判据（三条同时满足才算支持假说）：")
    log("  1) BV 的 r <= -0.30 且 p < 0.05")
    log("  2) BV 比 MAT 至少再负 0.15（r_bv - r_mat <= -0.15）")
    log("  3) 分组均值里 BV 随压缩倍数单调不增")
    gb = [mean([r["BV"] for r in rec if pred(r["factor"])]) for _, pred in buckets]
    gb = [v for v in gb if v == v]
    mono = all(gb[i] >= gb[i + 1] for i in range(len(gb) - 1)) if len(gb) > 1 else False

    c1 = (r_bv == r_bv) and r_bv <= -0.30 and p_bv < 0.05
    c2 = (r_bv == r_bv) and (r_mat == r_mat) and (r_bv - r_mat) <= -0.15
    log("")
    log("  1) %s   (r_bv=%+.3f, p=%.4f)" % ("✅" if c1 else "❌", r_bv, p_bv))
    log("  2) %s   (r_bv-r_mat=%+.3f)" % ("✅" if c2 else "❌", r_bv - r_mat))
    log("  3) %s   (分组 BV: %s)" % ("✅" if mono else "❌",
                                    " ".join("%.3f" % v for v in gb)))
    if c1 and c2:
        log("")
        log("  ✅ 支持假说 -> 下一步：改成论文协议（原生分辨率裁 512 训练 + 滑窗推理），")
        log("     重建 processed 并在**同一折**上与现有 `f0_axfix` 对比。")
    else:
        log("")
        log("  ❌ 不支持 -> 分辨率不是 BV 缺口的主因，别动预处理；")
        log("     转去查 BV 相关的配方（loss 权重 / 前景采样 / 增强强度）。")
    log("-" * 80)

    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[written] {out}")

    if args.csv:
        import csv as _csv
        cp = Path(os.path.expanduser(args.csv))
        with open(cp, "w", newline="", encoding="utf-8-sig") as fh:
            w = _csv.writer(fh)
            w.writerow(["core", "long_side", "factor", "BV", "MAT"])
            for r in rec:
                w.writerow([r["core"], r["long"], f"{r['factor']:.3f}",
                            "" if r["BV"] != r["BV"] else f"{r['BV']:.4f}",
                            "" if r["MAT"] != r["MAT"] else f"{r['MAT']:.4f}"])
        print(f"[csv] {cp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
