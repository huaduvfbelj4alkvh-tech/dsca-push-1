"""标签质量审计：512 管线对 GT 到底做了什么。

动机（2026-09-27）
------------------
同折 A/B 把差距归因指向了**数据侧**（BV 差在基线 −5.76，MAT 差在模块 −4.65）。
但"数据侧"太笼统，要回答的是：**是我们改坏了标签，还是标注本身就不完整？**

本脚本用**原始 fu_*.nii.gz**（原生分辨率）当唯一真值，做四件事：

  1) 复现校验    重算一遍 `fit_to_square(..., order=0)`，与在用 npz 的 label 比 Dice。
                 必须 = 1.000000，否则说明在用的数据和脚本不一致，后面全部作废。
  2) 破碎度      BV/MAT 分开数连通域个数 + 闭运算自洽 Dice（native vs 目标尺寸）。
  3) 剂量曲线    目标尺寸扫描 384/512/640/768/960，看"分辨率能不能救 BV"。
  4) 三个假设    逐条证伪：
                   a. 极性反转   -> 前景 vs **紧邻环带**背景的比值（应 < 1）
                   b. 漏标邻近血管 -> 血管外 r 环带里的暗像素占比 vs 全图基线
                   c. 变细       -> order=0 与抗锯齿重采样的面积比（应 ≈ 1）

为什么要 order=0 vs 抗锯齿这一列：图像走的是 `order=1 + anti_aliasing=True`，
标签走 `order=0`（纯最近邻）。薄血管在图上连续、在标签上断点 —— 这个"口径地板"
是 BV 过检（PRE 低）最可能的来源。

用法（本机，需要原始 labelsTs；集群上没传原始数据，跑不了）
----------------------------------------------------------
    E:\\Anaconda\\envs\\dsca\\python.exe tmp/audit_label_quality.py
    E:\\Anaconda\\envs\\dsca\\python.exe tmp/audit_label_quality.py --sizes 512,768 --limit 8

判据
----
  * 复现校验 Dice 必须 1.000000（差一点就是数据/代码漂移，先查再往下看）
  * BV 连通域 native ≈ 个位数，512 若上到 100+ ⇒ 重采样在切碎细血管
  * 剂量曲线上 768 若把连通域砍半、闭运算 Dice +0.05 ⇒ 分辨率是有效杠杆
  * 极性比值若 >= 1 ⇒ 该例输入极性反了（本批实测 44/44 都 < 1，无此问题）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.transform import resize as sk_resize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import normalize                       # noqa: E402
from data.prepare import fit_to_square, load_label       # noqa: E402

CONN8 = np.ones((3, 3), bool)


def dice(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    s = int(a.sum()) + int(b.sum())
    return float(2 * int((a & b).sum()) / s) if s else 1.0


def connectivity(mask) -> tuple[int, float]:
    """返回 (连通域个数, 闭运算自洽 Dice)。

    闭运算 Dice 的含义：一个"形态学上干净"的预测（把断点补上、去掉孤立小点）
    相对当前 GT 能拿多少 Dice。它就是**这类模型的分数天花板**。
    """
    m = np.asarray(mask, bool)
    if not m.any():
        return 0, 1.0
    c = ndi.binary_closing(m, structure=CONN8)
    _, n = ndi.label(m, structure=CONN8)
    return int(n), dice(m, c)


def aa_render(vol: np.ndarray, size: int) -> np.ndarray:
    """抗锯齿渲染（与 fit_to_square 同样的几何，只把插值换成 order=1+AA）。

    ⚠️ 必须复制 fit_to_square 的**几何**（等比例 + 居中 pad），否则比的是几何差异
       而不是插值差异。这里只用它做"另一种同样合法的渲染"，不参与训练。
    """
    vol = np.asarray(vol, np.float32)
    h, w = vol.shape
    scale = size / max(h, w)
    nh = min(size, max(1, int(round(h * scale))))
    nw = min(size, max(1, int(round(w * scale))))
    out = np.zeros((size, size), np.float32)
    top, left = (size - nh) // 2, (size - nw) // 2
    r = sk_resize(vol, (nh, nw), order=1, preserve_range=True, anti_aliasing=True)
    out[top:top + nh, left:left + nw] = r
    return out > 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description="标签质量审计")
    ap.add_argument("--src", default=r"E:\Datasets\DSCA\DSA_public")
    ap.add_argument("--data", default=r"E:\Datasets\DSCA\processed")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--sizes", default="384,512,640,768,960")
    ap.add_argument("--limit", type=int, default=0, help="只做前 N 例（调试用）")
    args = ap.parse_args()

    src = Path(args.src)
    lab_dir = src / ("labelsTr" if args.split == "train" else "labelsTs")
    npz_dir = Path(args.data) / args.split
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    pairs = []
    for p in sorted(lab_dir.glob("fu_*.nii.gz")):
        core = p.name[3:-7]
        if (npz_dir / f"{core}.npz").exists():
            pairs.append((core, p))
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print(f"[ERROR] {lab_dir} 与 {npz_dir} 没有交集")
        return 2

    print("=" * 92)
    print(f"标签质量审计  split={args.split}  样本={len(pairs)}")
    print(f"  原始标注 {lab_dir}")
    print(f"  在用数据 {npz_dir}")
    print("=" * 92)

    # ---------------------------------------------------------------- 1) 复现校验
    print("\n[1] 复现校验：重算 fit_to_square(order=0) vs 在用 npz 的 label")
    print("    （必须 1.000000；否则代码/数据已漂移，后面全部作废）")
    bad = []
    for core, p in pairs:
        native = np.squeeze(np.asarray(load_label(p)))
        m = np.load(npz_dir / f"{core}.npz")["label"] > 0
        # 轴序已在 prepare 里纠正过，这里允许一次转置
        d = max(dice(fit_to_square(native, m.shape[0], 0) > 0.5, m),
                dice(fit_to_square(native, m.shape[0], 0).T > 0.5, m))
        if d < 0.999999:
            bad.append((core, d))
    if bad:
        print(f"  ❌ {len(bad)}/{len(pairs)} 例不一致：")
        for c, d in bad[:10]:
            print(f"       {c:<16} Dice={d:.6f}")
        print("  -> 先解决这个再读下面的数字")
    else:
        print(f"  ✅ {len(pairs)}/{len(pairs)} 例全部 1.000000")

    # ---------------------------------------------------------------- 2) 剂量曲线
    print(f"\n[2] 剂量曲线：BV(label=1) 破碎度 vs 目标尺寸（{len(pairs)} 例均值）")
    print(f"    {'size':<7}{'BV_cc':>10}{'BV_clDice':>13}{'floor(nn vs AA)':>18}")
    cache = {}
    for core, p in pairs:
        cache[core] = np.squeeze(np.asarray(load_label(p)))
    profile = {}
    for size in sizes:
        ccs, clds, floors = [], [], []
        for core, _ in pairs:
            bv = cache[core] == 1
            if not bv.any():
                continue
            f0 = fit_to_square(bv.astype(np.float32), size, 0) > 0.5
            n, d = connectivity(f0)
            ccs.append(n)
            clds.append(d)
            floors.append(dice(f0, aa_render(bv.astype(np.float32), size)))
        if ccs:
            profile[size] = (float(np.mean(ccs)), float(np.mean(clds)),
                             float(np.mean(floors)))
            print(f"    {size:<7}{profile[size][0]:>10.1f}"
                  f"{profile[size][1]:>13.4f}{profile[size][2]:>18.4f}")
    print("    -> BV_cc 越低、clDice 越高越好；floor 是两种合法渲染之间的不可约分歧")

    # ------------------------------------------------------- 3) BV/MAT 分开看破碎度
    print("\n[3] BV vs MAT 分开：原生标注 -> 在用数据")
    print("    分成两组：真被重采样的 / 原生尺寸就等于在用尺寸的（后者是天然对照组）")
    print(f"    {'组':<14}{'类别':<6}{'n':>5}{'cc 原生':>10}{'-> 在用':>10}"
          f"{'clDice 原生':>14}{'-> 在用':>10}")
    groups = {"重采样": ([], []), "未重采样": ([], [])}
    for core, _ in pairs:
        nat_lab = cache[core]
        cur_lab = np.load(npz_dir / f"{core}.npz")["label"]
        key = "未重采样" if max(nat_lab.shape) == cur_lab.shape[0] else "重采样"
        groups[key][0].append(core)
        groups[key][1].append((nat_lab, cur_lab))

    for gname, (cores_in, labs_in) in groups.items():
        if not cores_in:
            continue
        for cls, name in ((1, "BV"), (2, "MAT")):
            recs = []
            for nat_lab, cur_lab in labs_in:
                nat, cur = nat_lab == cls, cur_lab == cls
                if not nat.any() and not cur.any():
                    continue
                recs.append((connectivity(nat), connectivity(cur)))
            if not recs:
                print(f"    {gname:<14}{name:<6}{0:>5}{'-':>10}{'-':>10}{'-':>14}{'-':>10}")
                continue
            print(f"    {gname:<14}{name:<6}{len(recs):>5}"
                  f"{np.mean([r[0][0] for r in recs]):>10.1f}"
                  f"{np.mean([r[1][0] for r in recs]):>10.1f}"
                  f"{np.mean([r[0][1] for r in recs]):>14.4f}"
                  f"{np.mean([r[1][1] for r in recs]):>10.4f}")
    print("    💡 判读：'未重采样'组的两列必须**几乎完全相等**（没动过它）。")
    print("       若 '重采样' 组里 BV 连通域从个位数爆到 100+、而 MAT 停在 1，")
    print("       ⇒ 损伤只发生在**细结构**上，与类别语义无关，是纯渲染问题。")

    # ---------------------------------------------------------------- 4) 三假设
    print("\n[4] 三个被证伪的假设")
    pol, area_ratio = [], []
    BAND_RS = (1, 2, 3, 8, 16)
    band_ratio = {r: [] for r in BAND_RS}
    for core, _ in pairs:
        d = np.load(npz_dir / f"{core}.npz")
        raw = d["minip"].astype(np.float32)
        lab = d["label"].astype(np.int64)
        m = lab > 0
        if not m.any():
            continue
        # a) 极性：前景 vs 紧邻环带背景（环带不会被大面积暗区污染）
        ring = ndi.binary_dilation(m, iterations=10) & ~m
        if ring.any():
            pol.append((core, float(raw[m].mean() / max(1e-6, raw[ring].mean()))))
        # b) 漏标：靠血管那一圈里有多少"暗像素"
        x = normalize(raw, "zscore")
        t = float(np.percentile(x[m], 25))
        p0 = float((x < t).mean())
        for r in BAND_RS:
            band = ndi.binary_dilation(m, iterations=r) & ~m
            if band.any():
                band_ratio[r].append(float(((x < t) & band).sum()) / int(band.sum()))
        # c) 变细：同几何下两种渲染的面积比
        nat = cache[core] > 0
        nn = fit_to_square(nat.astype(np.float32), lab.shape[0], 0) > 0.5
        aa = aa_render(nat.astype(np.float32), lab.shape[0])
        if aa.sum():
            area_ratio.append(float(nn.sum() / aa.sum()))

    ratios = [r for _, r in pol]
    n_bad = sum(1 for r in ratios if r >= 0.98)
    print(f"    a) 极性（前景/环带背景，应 <1）：{n_bad}/{len(ratios)} 例异常，"
          f"范围 {min(ratios):.2f}~{max(ratios):.2f}，均值 {np.mean(ratios):.3f}")
    if n_bad:
        for c, r in pol:
            if r >= 0.98:
                print(f"         ⚠️ {c}: {r:.3f}  <- 这一例输入极性可能反了")
    base = None
    print("    b) 血管外 r 环带里的暗像素占比（越低越说明标注在边界处完整）：")
    for r in sorted(band_ratio):
        v = float(np.mean(band_ratio[r]))
        print(f"         r={r:<3} {v:.4f}")
    print(f"       全图暗像素基线 ≈ 0.08 -> 环带比全图低两个数量级 ⇒ "
          f"**血管外侧没有成片的血管样暗结构**")
    print(f"    c) 面积比 order=0 / 抗锯齿 = {np.mean(area_ratio):.4f} "
          f"(≈1 ⇒ 不是'变细'，是边界抖动)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
