"""把已有的 eval_*.json 按两种「缺失类」口径重算，并给出论文口径的对照数字。

为什么需要它（2026-09-26 结论）
--------------------------------------------------------------
DSCA 官方标注里有 **8/44 张测试图根本没有 MAT（class 2）**。
已对源头 `DSA_public/labelsTs/fu_*.nii.gz` 逐例核实：raw 与 npz 的
class1/class2 体素数在未降采样样本上**逐位相同**，降采样样本严格按面积比缩放
⇒ 标签链忠实，缺 MAT 是官方标注的性质，不是我们预处理弄丢的。

「GT 里没有这个类，Dice 怎么算」直接决定最终数字，两种口径：

  union 口径（本仓库 metrics.py 的默认做法）
      标签有 OR 预测有 -> 计入平均
        · 漏检（标签有、预测无）   -> dice ≈ 0，计入
        · 幻觉（标签无、预测有）   -> dice ≈ 0，计入   <-- 比 label 口径多这一项惩罚
      关系：union 的分母 = label 的分母 + 幻觉样本数，两者**分子完全相同**。

  label 口径（判定为论文口径）
      只看「标签里确实有」的样本；GT 没有该类的图一律不进平均。

判定论文用 label 口径的依据（可复算）：
  Table III 中 CENet 的 MAT Dice = 84.32%，而 union 口径下 44 例里只有 36 例有 MAT，
  上限 = 36/44 = 81.8% —— 84.32 在 union 口径下**数学上不可能出现**；
  DSANet 的 92.26% 更不可能。故论文必然没有把「GT 无 MAT」的样本记成 0。

用法
--------------------------------------------------------------
    # 在集群上，一次把 5 折 + 各种 run 全算完
    python eval_calibrate.py --data ~/dsca_data/processed \
        --glob 'runs/*/eval_test.json' --out runs/calibration_summary.json

    # 只看某一个
    python eval_calibrate.py --data ~/dsca_data/processed \
        --glob 'runs/f0_miniponly/eval_test.json'
"""
from __future__ import annotations

import argparse
import csv
import glob as globmod
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

from metrics import CLASS_NAMES            # noqa: E402

FG_CLASSES = ("BV", "MAT")                 # 论文主报的「两类均」就是这两个


# ----------------------------------------------------------------------
def _mean(vals: list) -> tuple[float, int]:
    """跳过 None 和 NaN 求平均，返回 (mean, n)。"""
    keep = [float(v) for v in vals if v is not None and v == v]
    if not keep:
        return float("nan"), 0
    return sum(keep) / len(keep), len(keep)


def _fg_mean(per_class: dict) -> float:
    vals = [per_class[c] for c in FG_CLASSES
            if per_class.get(c) is not None and per_class[c] == per_class[c]]
    return sum(vals) / len(vals) if vals else float("nan")


class LabelPresence:
    """按需读取 npz 里的 label，缓存「每个类是否存在」。"""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def present(self, split: str, core: str) -> np.ndarray | None:
        sub = "test" if split == "test" else "train"
        key = (sub, core)
        if key not in self._cache:
            p = self.data_root / sub / f"{core}.npz"
            if not p.exists():
                self._cache[key] = None            # type: ignore[assignment]
            else:
                lab = np.load(p)["label"]
                self._cache[key] = np.array(
                    [(lab == c).sum() > 0 for c in range(len(CLASS_NAMES))])
        return self._cache[key]


# ----------------------------------------------------------------------
def calibrate_one(js: dict, lp: LabelPresence) -> dict:
    split = js.get("split", "test")
    ps = js.get("per_sample", [])
    out = {
        "run": Path(js.get("ckpt", "?")).parent.name or "?",
        "ckpt": js.get("ckpt"),
        "epoch": js.get("epoch"),
        "split": split,
        "tta": js.get("tta"),
        "n_samples": len(ps),
    }

    # ---- union：per_sample 里非 None 的就是「计入」的
    union_lists = {c: [] for c in CLASS_NAMES}
    for s in ps:
        for c in CLASS_NAMES:
            union_lists[c].append(s.get(c))

    # ---- label：再额外要求「标签里确实有」
    label_lists = {c: [] for c in CLASS_NAMES}
    gt_absent = {c: 0 for c in CLASS_NAMES}
    unknown = 0
    for s in ps:
        core = s.get("core")
        pr = lp.present(split, core) if core else None
        if pr is None:
            unknown += 1
            for c in CLASS_NAMES:
                label_lists[c].append(None)
            continue
        for i, c in enumerate(CLASS_NAMES):
            if pr[i]:
                label_lists[c].append(s.get(c))
            else:
                gt_absent[c] += 1
                label_lists[c].append(None)       # GT 没有 -> 不进平均

    union_pc, union_n = {}, {}
    label_pc, label_n = {}, {}
    for c in CLASS_NAMES:
        union_pc[c], union_n[c] = _mean(union_lists[c])
        label_pc[c], label_n[c] = _mean(label_lists[c])

    out["union_per_class"] = union_pc
    out["union_n"] = union_n
    out["union_fg"] = _fg_mean(union_pc)
    out["label_per_class"] = label_pc
    out["label_n"] = label_n
    out["label_fg"] = _fg_mean(label_pc)
    out["gt_absent"] = gt_absent
    out["n_unknown_gt"] = unknown

    # ---- 幻觉 / 漏检计数（解释两种口径差在哪）
    halluc = {c: 0 for c in CLASS_NAMES}
    miss = {c: 0 for c in CLASS_NAMES}
    for s in ps:
        core = s.get("core")
        pr = lp.present(split, core) if core else None
        if pr is None:
            continue
        for i, c in enumerate(CLASS_NAMES):
            v = s.get(c)
            if v is None or v != v:
                continue                          # 未计入
            if v > 0.01:
                continue
            if pr[i]:
                miss[c] += 1                      # 标签有、预测≈0 -> 真漏检
            else:
                halluc[c] += 1                    # 标签无、预测≈0 -> 幻觉被罚
    out["hallucination"] = halluc
    out["miss"] = miss

    # union 口径下 MAT 的理论上限
    n = out["n_samples"]
    if n and not unknown:
        out["union_mat_upper"] = (n - gt_absent["MAT"]) / n
    else:
        out["union_mat_upper"] = float("nan")
    return out


# ----------------------------------------------------------------------
def report(rows: list[dict]) -> None:
    print("=" * 96)
    for r in rows:
        print(f"run = {r['run']:<28} epoch={r['epoch']}  split={r['split']}  "
              f"tta={r['tta']}  n={r['n_samples']}"
              + (f"  [GT 缺失未知 {r['n_unknown_gt']} 例]" if r['n_unknown_gt'] else ""))
        upc, lpc = r["union_per_class"], r["label_per_class"]
        print(f"  {'口径':<8}{'BV':>18}{'MAT':>18}{'两类均':>12}")
        print(f"  {'union':<8}{upc['BV']:>10.4f} (n={r['union_n']['BV']:>2})"
              f"{upc['MAT']:>10.4f} (n={r['union_n']['MAT']:>2})"
              f"{r['union_fg']:>12.4f}")
        print(f"  {'label':<8}{lpc['BV']:>10.4f} (n={r['label_n']['BV']:>2})"
              f"{lpc['MAT']:>10.4f} (n={r['label_n']['MAT']:>2})"
              f"{r['label_fg']:>12.4f}")
        ga, ha, ms = r["gt_absent"], r["hallucination"], r["miss"]
        print(f"  GT 缺 MAT {ga['MAT']} 例 / 缺 BV {ga['BV']} 例"
              f"  |  其中被当 0 罚的幻觉: MAT {ha['MAT']} / BV {ha['BV']}"
              f"  |  真漏检: MAT {ms['MAT']} / BV {ms['BV']}")
        if r["union_mat_upper"] == r["union_mat_upper"]:
            print(f"  ⚠ union 口径下 MAT 的理论上限 = {r['union_mat_upper']*100:.1f}%"
                  f"  (论文 Table III CENet=84.32% 已超过它 -> 论文必非 union 口径)")
        print("-" * 96)

    if len(rows) > 1:
        fgs = [r["union_fg"] for r in rows if r["union_fg"] == r["union_fg"]]
        if fgs and min(fgs) < 0.5 < max(fgs):
            print("  🔴 警告：参与平均的 run 里同时存在 两类均<0.5 与 >0.5 的 ——"
                  "极可能混入了**不同数据版本**（如轴序 bug 期的 run）。\n"
                  "     这种混合下「跨 run 均值」没有意义，请加 --only 限定同类 run"
                  "（例：--only 'axfix'）。")
        for key, tag in (("union_fg", "union"), ("label_fg", "label")):
            vals = [r[key] for r in rows if r[key] == r[key]]
            if vals:
                a = np.array(vals)
                print(f"  {tag:<6} 两类均 跨 {len(a)} 个 run: "
                      f"mean {a.mean():.4f}  std {a.std(ddof=1) if len(a) > 1 else 0:.4f}"
                      f"   min {a.min():.4f}  max {a.max():.4f}")
        for key, tag in (("union_per_class", "union"), ("label_per_class", "label")):
            for c in FG_CLASSES:
                vals = [r[key][c] for r in rows if r[key][c] == r[key][c]]
                if vals:
                    a = np.array(vals)
                    print(f"  {tag:<6} {c:<4} 跨 {len(a)} 个 run: "
                          f"mean {a.mean():.4f}  std {a.std(ddof=1) if len(a) > 1 else 0:.4f}")
        print("=" * 96)


def main() -> int:
    ap = argparse.ArgumentParser(description="按两种缺类口径重算已有 eval json")
    ap.add_argument("--data", default=r"E:\Datasets\DSCA\processed",
                    help="processed 根目录（集群上必须显式传 ~/dsca_data/processed）")
    ap.add_argument("--glob", default="runs/*/eval_test.json",
                    help="eval json 的 glob（相对当前目录）")
    ap.add_argument("--only", default=None,
                    help="只保留**路径**匹配该正则的 eval json。"
                         "⚠️ 强烈建议用：集群 runs/ 下常同时躺着轴序 bug 期的 run"
                         "（全部锁在 0.13~0.18），混进跨 run 平均会得到毫无意义的数。"
                         "例：--only 'axfix'")
    ap.add_argument("--out", default=None, help="汇总 json 输出路径")
    ap.add_argument("--csv", default=None, help="汇总 csv 输出路径")
    args = ap.parse_args()

    data_root = Path(os.path.expanduser(args.data))
    if not data_root.exists():
        print(f"[ERROR] --data 不存在: {data_root}\n"
              f"        集群上请显式传 --data ~/dsca_data/processed")
        return 2

    files = sorted(globmod.glob(args.glob))
    if args.only:
        rx = re.compile(args.only)
        before = len(files)
        files = [f for f in files if rx.search(f)]
        print(f"--only {args.only!r} 过滤: {before} -> {len(files)} 个")
    if not files:
        print(f"[ERROR] 没匹配到任何 json: {args.glob}"
              + (f"（--only {args.only!r} 之后为空）" if args.only else "")
              + "\n        先看有哪些 run: ls -1 runs/")
        return 2

    lp = LabelPresence(data_root)
    rows = []
    for f in files:
        js = json.loads(Path(f).read_text(encoding="utf-8"))
        if not js.get("per_sample"):
            print(f"[WARN] {f} 里没有 per_sample，跳过")
            continue
        r = calibrate_one(js, lp)
        r["file"] = f
        rows.append(r)

    if not rows:
        print("[ERROR] 没有可用的结果")
        return 2

    print(f"扫描到 {len(rows)} 个 eval json（--data={data_root}）")
    report(rows)

    if args.out:
        out_p = Path(os.path.expanduser(args.out))
        out_p.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                         encoding="utf-8")
        print(f"汇总已写: {out_p}")
    if args.csv:
        with open(os.path.expanduser(args.csv), "w", newline="",
                  encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "epoch", "split", "n",
                        "union_BV", "union_MAT", "union_fg",
                        "label_BV", "label_MAT", "label_fg",
                        "gt_absent_MAT", "halluc_MAT", "miss_MAT"])
            for r in rows:
                w.writerow([r["run"], r["epoch"], r["split"], r["n_samples"],
                            f"{r['union_per_class']['BV']:.4f}",
                            f"{r['union_per_class']['MAT']:.4f}",
                            f"{r['union_fg']:.4f}",
                            f"{r['label_per_class']['BV']:.4f}",
                            f"{r['label_per_class']['MAT']:.4f}",
                            f"{r['label_fg']:.4f}",
                            r["gt_absent"]["MAT"], r["hallucination"]["MAT"],
                            r["miss"]["MAT"]])
        print(f"CSV 已写: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
