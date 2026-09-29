"""用老大贴回来的 16 个 run 输出，算准 axfix 5 折的两种口径均值/标准差，并做差距拆解。

不重跑模型——数据直接从 eval_calibrate.py 的打印结果抄过来（可对照 runs/calibration_summary.csv）。
"""
from __future__ import annotations

import statistics as st

# ---- 新数据（axfix）：唯一有效的一组 ----
# run: (epoch, tta, BV, MAT_union, MAT_label, fg_union, fg_label, MAT_n_union, halluc, miss)
AXFIX = {
    "f0_axfix": (220, True, 0.8146, 0.8390, 0.8856, 0.8268, 0.8501, 38, 2, 0),
    "f1_axfix": (320, True, 0.8202, 0.8242, 0.8929, 0.8222, 0.8565, 39, 3, 0),
    "f2_axfix": (230, True, 0.8184, 0.8065, 0.8962, 0.8125, 0.8573, 40, 4, 0),
    "f3_axfix": (320, True, 0.8187, 0.8516, 0.8989, 0.8351, 0.8588, 38, 2, 1),
    "f4_axfix": (230, True, 0.8144, 0.8893, 0.8893, 0.8518, 0.8518, 36, 0, 1),
}
MINIPONLY = (130, True, 0.7926, 0.7378, 0.9017, 0.7652, 0.8472, 44, 8, 0)

# ---- 旧数据（轴序 bug 期）：全部 ~0.13~0.19，作为"作废簇"留档 ----
LEGACY_UNION = {
    "5fold_adam_f0": 0.1587, "5fold_adam_f1": 0.1726, "5fold_adam_f2": 0.1660,
    "5fold_adam_f3r": 0.1763, "5fold_adam_f4": 0.1639, "adam500": 0.1660,
    "exp": 0.1544, "lc_n108": 0.1645, "lc_n36": 0.1328, "lc_n72": 0.1372,
}

# 论文锚点（两类均口径，Table III / V）
PAPER_MINIPONLY = (86.45, 86.22, 86.34)      # BV, MAT, 两类均
PAPER_FULL = (87.32, 92.26, 89.79)


def stat(vals):
    m = st.mean(vals)
    sd1 = st.stdev(vals) if len(vals) > 1 else 0.0     # ddof=1
    sd0 = st.pstdev(vals) if len(vals) > 1 else 0.0
    return m, sd1, sd0, min(vals), max(vals)


print("=" * 84)
print("1) DSANet 5 折（新数据 axfix）—— 两种口径")
print("=" * 84)
for tag, idx in (("两类均", 5), ("两类均", 6)):
    pass
cols = {
    "BV":            [v[2] for v in AXFIX.values()],
    "MAT(union)":    [v[3] for v in AXFIX.values()],
    "MAT(label)":    [v[4] for v in AXFIX.values()],
    "两类均 union":  [v[5] for v in AXFIX.values()],
    "两类均 label":  [v[6] for v in AXFIX.values()],
}
for k, v in cols.items():
    m, sd1, sd0, lo, hi = stat(v)
    print(f"  {k:<14} mean {m:.4f}  std(ddof=1) {sd1:.4f}  std(ddof=0) {sd0:.4f}  "
          f"min {lo:.4f}  max {hi:.4f}")

print()
print("  逐折（看分布，别只看均值）:")
print(f"  {'run':<10}{'ep':>5}{'BV':>9}{'MAT_u':>9}{'MAT_l':>9}{'fg_u':>9}{'fg_l':>9}"
      f"{'nMAT_u':>8}{'幻觉':>6}{'漏检':>6}")
for k, v in AXFIX.items():
    print(f"  {k:<10}{v[0]:>5}{v[2]:>9.4f}{v[3]:>9.4f}{v[4]:>9.4f}{v[5]:>9.4f}"
          f"{v[6]:>9.4f}{v[7]:>8}{v[8]:>6}{v[9]:>6}")

print()
print("=" * 84)
print("2) 恒等式自检：union MAT × n_union == label MAT × 36")
print("=" * 84)
for k, v in list(AXFIX.items()) + [("f0_miniponly", MINIPONLY)]:
    a, b = v[3] * v[7], v[4] * 36
    print(f"  {k:<14} {v[3]:.4f}×{v[7]:>2} = {a:8.4f}   |   {v[4]:.4f}×36 = {b:8.4f}   "
          f"差 {a-b:+.4f}")

print()
print("=" * 84)
print("3) 与论文的差距（两类均口径，都是 label）")
print("=" * 84)
ours_fg = stat(cols["两类均 label"])[0] * 100
ours_bv = stat(cols["BV"])[0] * 100
ours_mat = stat(cols["MAT(label)"])[0] * 100
print(f"  我们（label 口径，5 折均值）   BV {ours_bv:.2f}  MAT {ours_mat:.2f}  两类均 {ours_fg:.2f}")
print(f"  论文 MinIP-only (Table V)      BV {PAPER_MINIPONLY[0]:.2f}  MAT {PAPER_MINIPONLY[1]:.2f}  "
      f"两类均 {PAPER_MINIPONLY[2]:.2f}")
print(f"  论文 DSANet 全模型 (Table III) BV {PAPER_FULL[0]:.2f}  MAT {PAPER_FULL[1]:.2f}  "
      f"两类均 {PAPER_FULL[2]:.2f}")
print()
print("  🔴 逐类拆解（这才是关键）:")
print(f"  BV : 我们 {ours_bv:.2f}  vs 论文基线 {PAPER_MINIPONLY[0]:.2f} → {ours_bv-PAPER_MINIPONLY[0]:+.2f}"
      f"   vs 论文全模型 {PAPER_FULL[0]:.2f} → {ours_bv-PAPER_FULL[0]:+.2f}")
print(f"  MAT: 我们 {ours_mat:.2f}  vs 论文基线 {PAPER_MINIPONLY[1]:.2f} → {ours_mat-PAPER_MINIPONLY[1]:+.2f}"
      f"   vs 论文全模型 {PAPER_FULL[1]:.2f} → {ours_mat-PAPER_FULL[1]:+.2f}")
print(f"  两类均: 我们 {ours_fg:.2f}  vs 论文基线 {PAPER_MINIPONLY[2]:.2f} → {ours_fg-PAPER_MINIPONLY[2]:+.2f}"
      f"   vs 论文全模型 {PAPER_FULL[2]:.2f} → {ours_fg-PAPER_FULL[2]:+.2f}")
print()
print("  ⇒ 差额几乎全在 BV，MAT 反而**超过**论文的 MinIP-only 基线。")

print()
print("=" * 84)
print("4) 旧数据（轴序 bug 期）作废簇 vs 新数据簇 —— 有没有中间地带")
print("=" * 84)
lg = sorted(LEGACY_UNION.values())
new = sorted(v[5] for v in AXFIX.values())
print(f"  旧簇 union 两类均: n={len(lg)}  min {lg[0]:.4f}  max {lg[-1]:.4f}")
print(f"  新簇 union 两类均: n={len(new)} min {new[0]:.4f}  max {new[-1]:.4f}")
print(f"  ⇒ 两簇之间的空档 = {new[0]-lg[-1]:.4f}（{lg[-1]:.4f} → {new[0]:.4f}），"
      f"16 个 run 里没有一个落在这段区间")
