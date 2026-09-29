"""训练诊断：为什么 loss 卡住、前景 Dice = 0。

分四个 stage：

  --stage data     纯 CPU，不需要 GPU。检查
                     1) 强度分布（反归一化回原始 DICOM 值，判断 /4095 合不合适）
                     2) 🔴 图像与标签的**对齐**检查（最关键）
                     3) 标签类别占比
                     4) 增强后前景还剩多少

  --stage projection  原始 8 帧序列 vs 各种投影算子的前景/背景对比度

  --stage scale     🔴 判别实验：前景学不出来，是"损失被类别不平衡压垮"
                    还是"输入里没有足够信息"？
                      上界参照：直接按最优灰度阈值二值化能拿多少 Dice
                      对照实验：小 U-Net 在 (归一化 × 损失) 网格上各过拟合 400 iter
                    ⚠️ 2026-09-20 修正：本 stage 第一版用"3 层 CNN + SGD lr0.05 + 200 iter"
                       得出"类别不平衡坍缩"，**这个结论是错的** —— 那是个测不出信号
                       的弱实验（3 层 CNN 感受野只有 7×7，训 200 步根本不够）。
                       换成小 U-Net + Adam + 400 iter 后，同样数据前景 Dice 能到 0.70~0.99。
                       结论随之改为：信息在输入里，**主因是跨样本 21.8 倍尺度差**。

  --stage overfit  单/多样本过拟合测试。判据：loss 应能压到 0.2 以下、前景 Dice 明显 > 0。
                   还学不会就是**结构性问题**（数据/损失/网络），不是"训得不够久"。
                   batch 多张图一起过拟合，是检验"跨样本一致性"的直接办法。

用法：
    E:\\Anaconda\\envs\\dsca\\python.exe diagnose.py --stage data
    E:\\Anaconda\\envs\\dsca\\python.exe diagnose.py --stage scale
    E:\\Anaconda\\envs\\dsca\\python.exe diagnose.py --stage overfit --cores CXL_AP_38,FXR_AP_25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.dataset import (DEFAULT_NORM, INTENSITY_DIV, NORM_MODES,   # noqa: E402
                          DSCADataset, normalize)
from metrics import dice_per_class                     # noqa: E402
from models.dsanet import DSANet                       # noqa: E402
from models.losses import DeepSupervisionLoss, inverse_frequency_weights  # noqa: E402


# ----------------------------------------------------------------------------
def stage_data(args) -> int:
    root = Path(args.data)
    ds = DSCADataset(root, "train", augment=False)
    print(f"训练集共 {len(ds)} 例\n")

    print("=" * 76)
    print("1) 强度分布 + 🔴 跨样本尺度差（决定该用哪种归一化）")
    print("=" * 76)
    mins, maxs, means, p99s, names = [], [], [], [], []
    for f in ds.files:                       # 全量扫，不只前 40 例
        d = np.load(f)["minip"]
        mins.append(float(d.min()))
        maxs.append(float(d.max()))
        means.append(float(d.mean()))
        p99s.append(float(np.percentile(d, 99)))
        names.append(f.stem)
    q = lambda a, p: float(np.percentile(a, p))
    print(f"  MinIP 原始值  min   : {q(mins,0):.0f}  {q(mins,50):.0f}  {q(mins,100):.0f}   (0/50/100 分位)")
    print(f"  MinIP 原始值  max   : {q(maxs,0):.0f}  {q(maxs,50):.0f}  {q(maxs,100):.0f}")
    print(f"  MinIP 原始值  mean  : {q(means,0):.0f}  {q(means,50):.0f}  {q(means,100):.0f}")

    spread = q(means, 100) / max(1e-6, q(means, 0))
    print(f"\n  🔴 跨样本 mean 的最大/最小 = {spread:.1f} 倍")
    dark = [(names[i], means[i], maxs[i]) for i in range(len(names))
            if means[i] < 0.2 * q(means, 50)]
    if dark:
        print(f"  疑似 8-bit 或另一套标定的样本 {len(dark)} 例（mean < 中位数的 20%）:")
        for nm, me, mx in sorted(dark, key=lambda t: t[1])[:12]:
            print(f"      {nm:<18} mean {me:>7.1f}   max {mx:>5.0f}   "
                  f"除以 4095 后最大仅 {mx/INTENSITY_DIV:.4f}")
        print(f"  → 这些样本在固定 /4095 下**近乎全黑**，和其余样本混训会让模型学不到一致规则。")
    if spread > 5:
        print(f"  → 尺度差 {spread:.1f} 倍 ⇒ **必须用 per-image 归一化**"
              f"（默认 {DEFAULT_NORM}）；--norm fixed 只作对照。")

    print()
    print("=" * 76)
    print("2) 🔴 图像与标签的对齐检查（最关键）")
    print("   血管在 DSA 中造影剂充盈 -> 像素偏暗，所以前景区的强度应**显著低于**背景")
    print("   （这里读的是 npz 里的原始像素，不经过归一化，才能直接比大小）")
    print("=" * 76)
    ratios, ring_ratios = [], []
    n = min(20, len(ds))
    for i in range(n):
        raw = np.load(ds.files[i])
        minip = raw["minip"].astype(np.float32)
        label = raw["label"].astype(np.int64)
        fg, bg = minip[label > 0], minip[label == 0]
        if fg.size == 0:
            continue
        r = float(fg.mean() / max(1e-6, bg.mean()))
        ratios.append(r)
        # 🔴 局部口径：紧邻环带背景。全图口径会被大面积暗区带偏（2026-09-27）
        r_loc = 1.0 - _ring_contrast(minip, label)
        ring_ratios.append(r_loc)
        flag = "✅" if r_loc < 1.0 else "❌ 前景比背景还亮"
        print(f"  {ds.cores[i]:<18} 前景均值 {fg.mean():>8.1f}  "
              f"全图背景 {bg.mean():>8.1f} 比值 {r:.3f}  |  "
              f"环带背景比值 {r_loc:.3f}  {flag}")
    if ratios:
        med = float(np.median(ratios))
        med_loc = float(np.median(ring_ratios))
        print(f"\n  全图口径比值中位数 = {med:.3f}（会被暗区污染，仅参考）")
        print(f"  🔴 环带口径比值中位数 = {med_loc:.3f}  <- **用这个判极性**（< 1 = 对齐正确）")
        if med_loc >= 1.0:
            print("  ❌ 前景比紧邻背景还亮：图像与标签很可能错位，或强度极性反了")
        elif med >= 1.0:
            print("  ⚠️ 全图口径 ≥1 但环带口径正常 -> 只是样本里有大面积暗区（准直器/未曝光区），"
                  "不是极性反了；per-image 归一化会被它带偏，注意即可。")

    print()
    print("=" * 76)
    print("3) 标签类别占比")
    print("=" * 76)
    hist = np.zeros(4, dtype=np.int64)
    for f in ds.files:
        d = np.load(f)
        v, c = np.unique(d["label"], return_counts=True)
        for vv, cc in zip(v.tolist(), c.tolist()):
            if vv < 4:
                hist[vv] += cc
    tot = hist.sum()
    print(f"  背景 {hist[0]/tot*100:.2f}%  BV {hist[1]/tot*100:.2f}%  "
          f"MAT {hist[2]/tot*100:.2f}%")
    print(f"  → 前景仅占 {(hist[1]+hist[2])/tot*100:.2f}%")

    print()
    print("=" * 76)
    print("4) 增强对前景的影响（增强把血管转出画面的比例）")
    print("=" * 76)
    base_ds = DSCADataset(root, "train", augment=False)
    aug_ds = DSCADataset(root, "train", augment=True, seed=123)
    for i in range(min(8, len(base_ds))):
        _, _, l0 = base_ds[i]
        _, _, l1 = aug_ds[i]
        a, b = (l0 > 0).float().mean().item(), (l1 > 0).float().mean().item()
        keep = b / max(1e-9, a)
        print(f"  {base_ds.cores[i]:<18} 原前景 {a*100:.2f}%  ->  增强后 {b*100:.2f}%  "
              f"保留 {keep*100:.0f}%")
    return 0


def _contrast(proj: np.ndarray, mask: np.ndarray) -> float:
    """前景(血管)与背景的对比度 = (背景均值 - 前景均值) / 背景均值。越大越好。

    ⚠️ 这里用的背景是**全图背景**，会被大面积暗区（准直器 / 未曝光区）拉低 —— 实测
       12/44 例约 27% 的面积是暗区，同一张图能算出 +0.13 也能算出 −0.04。
       判"输入极性对不对"请用下面的 `_ring_contrast`。本函数保留原义，只用于
       **同一批样本之间**比较不同投影算子（相对比较仍然有效）。
    """
    fg = proj[mask > 0].mean()
    bg = proj[mask == 0].mean()
    return float((bg - fg) / max(1e-6, bg))


def _ring_contrast(proj: np.ndarray, mask: np.ndarray, r: int = 10) -> float:
    """前景 vs **紧邻环带背景**的对比度（血管偏暗 -> 应为正）。

    环带 = 血管外 r 像素。它不会被大面积暗区污染，是判"极性是否正常"的可信口径。
    2026-09-27 实测：44/44 例都是 +0.47~+0.80 的正值。
    """
    from scipy import ndimage as ndi
    m = mask > 0
    if m.sum() == 0:
        return float("nan")
    ring = ndi.binary_dilation(m, iterations=r) & ~m
    if ring.sum() == 0:
        return float("nan")
    fg, bg = proj[m].mean(), proj[ring].mean()
    return float((bg - fg) / max(1e-6, bg))


def stage_projection(args) -> int:
    """检查 8 帧序列本身，以及各类投影算子的前景/背景对比度。

    动机：要确认三件事：
      1) 8 帧里每一帧各自的对比度（是不是有某帧特别清楚，MinIP 反而把它平均掉了）
      2) min / max / mean / median 哪种投影对比度最高
      3) DICOM 的 RescaleSlope / Intercept 要不要应用
         （实测不同样本的像素尺度相差 20 倍以上，固定 /4095 是错的）

    🔴 2026-09-27 口径修正：这里原来用**全图背景**均值（得出血管理论对比度只有 ~4%），
       那个数是被大面积暗区拉低后的假象。改用**紧邻环带背景**后，实测 44/44 例都是
       0.47~0.80 的正值 —— 输入信号其实很充足。本表已全部切到环带口径。
    """
    from data.prepare import (core_of, fit_to_square, fix_label_axis,   # noqa: E402
                              load_dicom_seq, load_label)

    src = Path(args.src)
    imgs = {core_of(p): p for p in sorted((src / "imagesTr").glob("*.dcm"))}
    labs = {core_of(p): p for p in sorted((src / "labelsTr").glob("*.nii.gz"))}
    pool = sorted(set(imgs) & set(labs))
    cores = args.cores.split(",") if args.cores else pool[: args.n]

    print("=" * 108)
    print("各帧与各投影算子的前景/背景对比度（环带口径，越大越好；负数=前景比背景亮，极性反了）")
    print("=" * 108)
    hdr = "样本".ljust(16) + "slope/inter".rjust(13) + "".join(
        f"帧{i}".rjust(7) for i in range(8)) + "".join(
        f"{s}".rjust(9) for s in ("min", "max", "mean", "median"))
    print(hdr)
    print("-" * 108)

    summary = {k: [] for k in ("min", "max", "mean", "median")}
    frame_scores = [[] for _ in range(8)]

    for core in cores:
        seq, meta = load_dicom_seq(imgs[core])
        lab = load_label(labs[core])
        # 🔴 2026-09-27 两处修正：
        #   1) fix_label_axis 返回 (label, 方向标记, 判定信息) 三个值，原来按 2 个解包
        #      -> `--stage projection` 一跑就 ValueError（这条路径一直没被执行过）。
        #   2) 必须传 ref_img：不传就只有"形状判据"可用，而形状判据在正方形样本上
        #      静默放行（196/224 错位），会让下面所有对比度数字是非位错位的结果。
        lab, _how, _ax = fix_label_axis(lab, meta["rows"], meta["cols"],
                                        ref_img=seq.min(axis=0), mode="align")
        if lab.shape != seq.shape[-2:]:
            seq = fit_to_square(seq, max(lab.shape), order=1)
            seq = seq[:, :lab.shape[0], :lab.shape[1]]
        if seq.shape[0] != 8:
            continue

        sc = [round(_ring_contrast(seq[i], lab), 3) for i in range(8)]
        for i, v in enumerate(sc):
            frame_scores[i].append(v)

        projs = {
            "min": seq.min(axis=0),
            "max": seq.max(axis=0),
            "mean": seq.mean(axis=0),
            "median": np.median(seq, axis=0),
        }
        ps = {k: _ring_contrast(v, lab) for k, v in projs.items()}
        for k in summary:
            summary[k].append(ps[k])

        row = core.ljust(16)
        row += f"{meta['slope']:.2g}/{meta['intercept']:.0f}".rjust(13)
        row += "".join(f"{v:>7.3f}" for v in sc)
        row += "".join(f"{ps[k]:>9.3f}" for k in ("min", "max", "mean", "median"))
        print(row)

    print("-" * 108)
    print("各帧平均对比度 : " + "".join(f"{np.mean(v):>7.3f}" for v in frame_scores))
    print("各投影平均对比度: " + "".join(f"{np.mean(summary[k]):>9.3f}"
                                        for k in ("min", "max", "mean", "median")))
    best = max(summary, key=lambda k: np.mean(summary[k]))
    print(f"\n  → 对比度最高的投影是 '{best}'（均值 {np.mean(summary[best]):.3f}）")
    if best != "min":
        print(f"  ⚠️ 当前代码用的是 MinIP，而上表显示 '{best}' 更好 -> 需要改投影算子")
    else:
        print("  ✅ MinIP 确实是当前最优投影")
    print("  → 环带口径下实测普遍在 0.4~0.8；若全都 < 0.20 才说明数据本身对比度低，"
          "需要更强的归一化（per-image）或改用减影思路")
    return 0


class _DoubleConv(torch.nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.b = torch.nn.Sequential(
            torch.nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            torch.nn.GroupNorm(8, cout), torch.nn.GELU(),
            torch.nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            torch.nn.GroupNorm(8, cout), torch.nn.GELU())

    def forward(self, x):
        return self.b(x)


class _SmallUNet(torch.nn.Module):
    """3 级小 U-Net。只做可学性判别，不追求性能。

    ⚠️ 必须用带 skip 的编解码结构，**不能用 3 层直筒 CNN**：
       直筒 CNN 感受野只有 7×7，看不到血管的细长结构，
       会在数据完全可学的情况下也给出 Dice=0 —— 就会得出错误结论。
    """

    def __init__(self, base=24, n_classes=3):
        super().__init__()
        c = [base, base * 2, base * 4]
        self.inp = _DoubleConv(1, c[0])
        self.d1 = _DoubleConv(c[0], c[1])
        self.d2 = _DoubleConv(c[1], c[2])
        self.u1 = torch.nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.b1 = _DoubleConv(c[1] * 2, c[1])
        self.u0 = torch.nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.b0 = _DoubleConv(c[0] * 2, c[0])
        self.head = torch.nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        import torch.nn.functional as F
        x0 = self.inp(x)
        x1 = self.d1(F.max_pool2d(x0, 2))
        x2 = self.d2(F.max_pool2d(x1, 2))
        y = self.b1(torch.cat([self.u1(x2), x1], 1))
        y = self.b0(torch.cat([self.u0(y), x0], 1))
        return self.head(y)


def _oracle_threshold(x: np.ndarray, label: np.ndarray) -> tuple[float, list[float]]:
    """上界参照：直接按最优灰度阈值二值化，最多能拿到多少 Dice。

    这一条回答的是"输入里到底有多少可用信号"，与网络无关。
    """
    fg = label > 0
    best_d, best_t = -1.0, 0.0
    for t in np.linspace(np.percentile(x, 0.5), np.percentile(x, 60), 120):
        p = x < t                       # 血管偏暗
        s = p.sum() + fg.sum()
        d = float(2 * (p & fg).sum() / s) if s else 0.0
        if d > best_d:
            best_d, best_t = d, float(t)
    # 两级近似：前景里更暗的那部分当 MAT
    p_fg = x < best_t
    dv = dm = float("nan")
    if p_fg.sum():
        tt = np.percentile(x[p_fg], 20)
        p_mat = (p_fg & (x < tt)).astype(np.int64)
        p_bv = (p_fg & ~(x < tt)).astype(np.int64)
        # 注意：dice_per_class 返回 (逐类 dice, 逐类是否出现) 二元组
        dv = float(dice_per_class(p_bv, label)[0][1])
        dm = float(dice_per_class(p_mat * 2, label)[0][2])
    return best_d, [float("nan"), dv, dm]


def stage_scale(args) -> int:
    """🔴 判别实验：是"损失的锅"还是"输入的锅"？

    做两件事：
      1) **上界参照**：直接把灰度按最优阈值二值化能拿多少 Dice。
         若这个数明显 > 0，说明信号在输入里；若它 ≈ 0，说明输入表示本身不够。
      2) **对照实验**：小 U-Net（带 skip）在 (归一化 × 损失) 网格上各过拟合 400 iter。
         哪一种组合能把前景 Dice 拉起来，就是该采用的配置。

    2026-09-20 实测结论（本机）：
      * oracle 阈值 Dice ≈ 0.19~0.21 → 逐像素信号很弱，但**不是零**
      * 小 U-Net + z-score + CE：BV 0.70 / MAT 0.93（CXL_AP_38 可达 0.99）
      * 同一张图，/4095 与 z-score 的单样本表现都还行
        → 单样本过得去、跨样本训不动 ⇒ **主因是跨样本尺度差（21.8 倍），不是损失**
    """
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data)
    cores = args.cores.split(",") if args.cores else ["CXL_AP_38", "FXR_AP_25"]
    iters = args.iters if args.iters != 200 else 400

    print("=" * 104)
    print("判别实验：前景学不出来 —— 是损失的锅，还是输入的锅？")
    print(f"  设备 {device}   iter/组合 = {iters}")
    print("=" * 104)

    losses = {
        "CE": lambda lg, tg, w: F.cross_entropy(lg, tg),
        "CE(加权)": lambda lg, tg, w: F.cross_entropy(lg, tg, weight=w),
        "Dice(不含背景)": lambda lg, tg, w: _dice_nobg(lg, tg),
        "CE_w+Dice": lambda lg, tg, w: F.cross_entropy(lg, tg, weight=w) + _dice_nobg(lg, tg),
    }

    for core in cores:
        path = root / "train" / f"{core}.npz"
        if not path.exists():
            print(f"[WARN] 找不到 {path}，跳过")
            continue
        d = np.load(path)
        raw = d["minip"].astype(np.float32)
        label = d["label"].astype(np.int64)
        freq = np.array([(label == c).mean() for c in range(3)])

        print()
        print("#" * 104)
        print(f"# 样本 {core}   raw min/max/mean = {raw.min():.0f}/{raw.max():.0f}/{raw.mean():.0f}"
              f"   类别占比 {freq[0]*100:.2f}/{freq[1]*100:.2f}/{freq[2]*100:.2f}%"
              f"   前景合计 {freq[1:].sum()*100:.2f}%")
        print("#" * 104)

        print("\n[上界参照] 直接把灰度按最优阈值二值化能拿到多少 Dice（与网络无关）")
        for m in ("fixed", "percentile", "zscore"):
            x = normalize(raw, m)
            d_all, dc = _oracle_threshold(x, label)
            print(f"  {m:<12} 前景整体 {d_all:.4f}   （两级近似）BV {dc[1]:.4f}  MAT {dc[2]:.4f}")
        print("  → 这个数就是「逐像素能拿到的天花板」；U-Net 靠空间结构可以远超它。")

        tgt = torch.from_numpy(label)[None].to(device)
        # 与 train.py 用同一套权重算法（含"保持损失尺度"的归一化）
        w = inverse_frequency_weights(freq, 3).to(device)

        print(f"\n[对照实验] 小 U-Net(3 级, base=24) × 3 归一化 × 4 损失，各 {iters} iter（Adam 1e-3）",
              flush=True)
        for m in ("fixed", "percentile", "zscore"):
            x = normalize(raw, m)
            inp = torch.from_numpy(x.astype(np.float32))[None, None].to(device)
            print()
            print(f"  {'归一化':<12}{'损失':<18}{'loss首':>9}{'loss末':>9}"
                  f"{'bg':>8}{'BV':>8}{'MAT':>8}")
            print("  " + "-" * 78)
            for lname, lfun in losses.items():
                torch.manual_seed(0)
                net = _SmallUNet().to(device).train()
                opt = torch.optim.Adam(net.parameters(), lr=1e-3)
                first = last = float("nan")
                for it in range(iters):
                    lg = net(inp)
                    loss = lfun(lg, tgt, w)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    if it == 0:
                        first = float(loss.detach())
                    last = float(loss.detach())
                with torch.no_grad():
                    pred = net(inp).argmax(1)[0].cpu().numpy()
                # dice_per_class 返回 (逐类 dice 数组, 逐类是否出现)
                dd = dice_per_class(pred, label)[0]
                print(f"  {m:<12}{lname:<18}{first:>9.4f}{last:>9.4f}"
                      f"{float(dd[0]):>8.4f}{float(dd[1]):>8.4f}{float(dd[2]):>8.4f}",
                      flush=True)

    print()
    print("=" * 104)
    print("判读规则（2026-09-20 修订版）")
    print("=" * 104)
    print("  1) 只要某一格 BV/MAT 明显 > 0  →  数据可学，问题在配置（归一化 / 损失 / 优化）")
    print("  2) 所有格都 ≈ 0               →  输入表示不够，要改输入（分辨率 / 投影 / 多帧）")
    print("  3) ⚠️ 单样本能学会 **不能** 推出跨样本也能学会。"
          "跨样本失败时优先怀疑尺度/分布不一致，")
    print("     用 --stage overfit --cores A,B,C,D 多张图一起过拟合来验。")
    return 0


def _dice_nobg(logits: torch.Tensor, target: torch.Tensor, n_classes: int = 3,
               eps: float = 1e-5) -> torch.Tensor:
    import torch.nn.functional as F
    probs = logits.softmax(1)
    oh = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    probs, oh = probs[:, 1:], oh[:, 1:]
    inter = (probs * oh).sum((2, 3))
    denom = probs.sum((2, 3)) + oh.sum((2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    present = oh.sum((2, 3)) > 0
    dice = torch.where(present, dice, torch.ones_like(dice))
    return 1.0 - dice.mean()


# ----------------------------------------------------------------------------
def stage_overfit(args) -> int:
    """单/多样本过拟合测试。

    ⭐ 传多个 --cores 时会把它们拼成一个 batch **一起**过拟合 —— 这是检验
       "跨样本一致性"的直接办法。如果单张能学会、多张学不会，
       那问题一定是样本间的分布/尺度不一致，而不是网络容量不够。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data)
    cores = (sorted(p.stem for p in (root / "train").glob("*.npz"))[:1]
             if args.cores is None else args.cores.split(","))
    ds = DSCADataset(root, "train", cores=cores, augment=False, norm=args.norm)

    seqs, minips, labels = [], [], []
    for c in cores:
        s, m, l = ds[ds.cores.index(c)]
        seqs.append(s)
        minips.append(m)
        labels.append(l)
    seq = torch.stack(seqs).to(device)          # (B,T,1,H,W)
    minip = torch.stack(minips).to(device)      # (B,1,H,W)
    label = torch.stack(labels).to(device)      # (B,H,W)

    print("=" * 78)
    print(f"过拟合测试  样本 {len(cores)} 例: {','.join(cores)}")
    print(f"  归一化 {args.norm}   优化器 {args.optimizer} lr={args.lr}   iter={args.iters}")
    print("  判据：loss 应能压到 0.2 以下、前景 Dice 明显 > 0")
    print("=" * 78)

    # 原始像素尺度，用来对照"跨样本尺度差有多大"
    for c in cores:
        raw = np.load(root / "train" / f"{c}.npz")["minip"]
        print(f"    原始 minip: {c:<16} min {raw.min():>5}  max {raw.max():>5}  "
              f"mean {raw.mean():>7.1f}")
    print()

    model = DSANet(num_frames=8, grad_checkpoint=args.grad_checkpoint).to(device).train()
    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99,
                              weight_decay=3e-5, nesterov=args.nesterov)
    cw = None
    if args.class_weights == "invfreq":
        cnt = np.bincount(label.cpu().numpy().ravel(), minlength=3).astype(np.int64)
        cw = inverse_frequency_weights(cnt, 3)
        print(f"  CE 类别权重 invfreq: {[round(float(x), 3) for x in cw]}")
    crit = DeepSupervisionLoss(
        n_classes=3,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        aux_weight=args.aux_weight,
        class_weights=cw,
        focal_gamma=args.focal_gamma,
    ).to(device)          # ⚠️ 必须 .to(device)：class_weights 是 buffer，
                          #    留在 CPU 会让 `class_weights[target]` 索引报设备不一致
    if cw is not None or args.focal_gamma or args.dice_weight != 1.0 or args.ce_weight != 1.0:
        print(f"  损失配置: CE×{args.ce_weight}  Dice×{args.dice_weight}  "
              f"aux×{args.aux_weight}  focal_gamma={args.focal_gamma}")
    print()

    for it in range(1, args.iters + 1):
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=args.amp and device.type == "cuda"):
            main, aux = model(seq, minip)
            loss, parts = crit(main, aux, label)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
        opt.step()

        if it % 25 == 0 or it == 1:
            with torch.no_grad():
                pred = main.float().argmax(1).cpu().numpy()
            dice = [dice_per_class(pred[i], label[i].cpu().numpy()) for i in range(len(cores))]
            detail = "  ".join(
                f"{cores[i]}: BV{v[0][1]:.3f}/MAT{v[0][2]:.3f}" for i, v in enumerate(dice))
            print(f"  iter {it:>4}  loss {float(loss.detach()):.4f}  "
                  f"(main {parts['loss_main']:.4f} aux {parts['loss_aux']:.4f})  {detail}",
                  flush=True)

    with torch.no_grad():
        pred = main.float().argmax(1).cpu().numpy()
    print()
    for i, c in enumerate(cores):
        v, n = np.unique(pred[i], return_counts=True)
        print(f"  {c:<16} 预测类别分布 {dict(zip(v.tolist(), [int(x) for x in n.tolist()]))}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="训练诊断")
    ap.add_argument("--stage", choices=["data", "projection", "scale", "overfit"],
                    default="data")
    ap.add_argument("--data", default=r"E:\Datasets\DSCA\processed")
    ap.add_argument("--src", default=r"E:\Datasets\DSCA\DSA_public",
                    help="原始数据目录（projection 阶段用）")
    ap.add_argument("--n", type=int, default=8, help="projection 阶段检查几个样本")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"],
                    help="过拟合测试用哪个优化器。sgd=与真实训练一致；"
                         "adam 收敛快，用于快速判断网络有没有容量学会")
    ap.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True,
                    help="SGD 的 Nesterov 动量（nnUNet 配方为 True）")
    ap.add_argument("--norm", default=DEFAULT_NORM, choices=list(NORM_MODES))
    ap.add_argument("--cores", default=None, help="逗号分隔的样本名，默认取前几个")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="过拟合测试用不用 AMP。排查'学不动'时把它关掉对比")
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=False,
                    help="过拟合测试默认关掉，跑得更快")
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--class-weights", default="none", choices=["none", "invfreq"],
                    help="invfreq = 按**所选样本**的类别频率取倒数加权 CE。"
                         "用于检验'背景是唯一一致方向'的假设")
    ap.add_argument("--focal-gamma", type=float, default=0.0,
                    help=">0 把 CE 换成 focal loss（压制易分的背景像素）")
    args = ap.parse_args()
    if args.stage == "data":
        return stage_data(args)
    if args.stage == "projection":
        return stage_projection(args)
    if args.stage == "scale":
        return stage_scale(args)
    return stage_overfit(args)


if __name__ == "__main__":
    sys.exit(main())
