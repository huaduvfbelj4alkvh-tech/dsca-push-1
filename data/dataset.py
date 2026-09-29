"""DSCA 数据集与数据增强。

增强项对照论文第 IV-A 节（随机旋转 / 弹性形变 / 随机缩放 / 随机裁切 / gamma / 镜像）：

    论文增强项        本实现
    ----------------  ------------------------------------------------
    镜像              有（水平 + 垂直，各 50% 概率）
    随机旋转          有（绕中心 ±15°，与缩放合并在同一次仿射变换里）
    随机缩放          有（0.9 ~ 1.1）
    gamma             有（0.7 ~ 1.5，只作用于图像，不动标签）
    弹性形变          有，但**默认关闭**（--elastic 打开）
                      原因：scipy 的 map_coordinates 对 8 帧 × 512² 每例要 ~250 ms，
                      数据加载会变成 batch=1 训练的瓶颈
    随机裁切          不适用：预处理已把所有图统一到 512×512，
                      没有比 512 更大的区域可裁（论文的原图分辨率更大）

⚠️ 三者必须用**同一套几何变换**（seq / minip / label），否则图和标签会错位，
   而这种错位不会报错，只会让 Dice 莫名偏低。

------------------------------------------------------------------
🔴 归一化：为什么默认不用 /4095（2026-09-20 实测后改的）
------------------------------------------------------------------
原始数据的像素尺度**跨样本相差 21.8 倍**（实测 60 例）：
    FXR_* / JYD_* 系列   min 0 / max 174~254 / mean ~125~145
    其余样本            min 0 / max 2800~4100 / mean ~2250~2700
固定除以 4095 会让 FXR/JYD 那一批整图落在 0~0.06，**几乎是全黑**，
而这些样本混在同一个训练集里。

为什么"单样本过拟合能学会、多样本训练学不会"：单样本时网络只要拟合这一张图的
灰度范围即可；一旦样本混在一起，**在一个样本上学到的阈值到另一个样本上全错**。
实测证据：小 U-Net 单样本 400 iter，z-score 下 BV Dice 0.70/0.99，
而固定 /4095 下跨样本训练 1000 iter 前景 Dice 恒为 0。

所以默认改成 **per-image z-score**，统计量在每张图自己身上算，把跨样本尺度差抹平。
`fixed` 模式保留下来做对照实验，不要删。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import affine_transform, gaussian_filter, map_coordinates
from torch.utils.data import Dataset

#: 原始 DICOM 的 12-bit 值域上限。**只给 fixed 模式用**，别拿它当默认归一化。
INTENSITY_DIV = 4095.0

#: 支持的归一化模式
NORM_MODES = ("zscore", "percentile", "minmax", "fixed")

#: 默认模式。理由见模块 docstring（全量实测：跨样本像素尺度差 25.5 倍）。
DEFAULT_NORM = "zscore"


def image_stats(x: np.ndarray, fg_thresh: float = 0.0) -> tuple[float, float]:
    """在"成像视野内"的像素上算 (mean, std)。

    DSA 图像圆形视野外是 0（准直器遮挡），这些像素既不是血管也不是组织，
    把它们算进均值/方差会把 std 抬虚。所以只在 x > fg_thresh 的像素上统计；
    若非零像素太少（<10%），退化回全图统计。
    """
    m = x > fg_thresh
    if m.mean() < 0.1:
        m = np.ones(x.shape, dtype=bool)
    vals = x[m]
    return float(vals.mean()), float(vals.std())


def normalize(x: np.ndarray, mode: str = DEFAULT_NORM,
              p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """把原始像素（uint16 量级）归一化成 float32。

    mode:
        zscore      (x - mean) / std，统计量取自本图视野内像素   <- 默认
        percentile  裁到 [p_lo, p_hi] 百分位后线性映射到 [0,1]
        minmax      按本图 min/max 线性映射到 [0,1]
        fixed       x / 4095，与样本无关（**仅作对照，跨样本尺度差会让它失效**）
    """
    x = x.astype(np.float32)
    if mode == "fixed":
        return x / INTENSITY_DIV
    if mode == "zscore":
        mu, sd = image_stats(x)
        return (x - mu) / (sd + 1e-6)
    if mode == "percentile":
        lo, hi = (float(v) for v in np.percentile(x, [p_lo, p_hi]))
    elif mode == "minmax":
        lo, hi = float(x.min()), float(x.max())
    else:
        raise ValueError(f"未知归一化模式 {mode!r}，可选 {NORM_MODES}")
    return np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def _apply_affine(vol: np.ndarray, matrix: np.ndarray, offset: np.ndarray,
                  order: int) -> np.ndarray:
    """对 (T,H,W) 或 (H,W) 施加同一个 2D 仿射变换。"""
    kw = dict(matrix=matrix, offset=offset, order=order, mode="constant", cval=0.0)
    if vol.ndim == 3:
        return np.stack([affine_transform(vol[i], **kw) for i in range(vol.shape[0])])
    return affine_transform(vol, **kw)


def _elastic(vol: np.ndarray, alpha: float, sigma: float, rng: np.random.RandomState,
             order: int) -> np.ndarray:
    """简化弹性形变：对随机位移场做高斯平滑后按位移采样。"""
    shape = vol.shape[-2:]
    dx = gaussian_filter(rng.uniform(-1, 1, shape), sigma, mode="reflect") * alpha
    dy = gaussian_filter(rng.uniform(-1, 1, shape), sigma, mode="reflect") * alpha
    yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    coords = [yy + dy, xx + dx]

    def warp(img: np.ndarray) -> np.ndarray:
        return map_coordinates(img, coords, order=order, mode="constant", cval=0.0)

    if vol.ndim == 3:
        return np.stack([warp(vol[i]) for i in range(vol.shape[0])])
    return warp(vol)


def _gamma(vol: np.ndarray, g: float) -> np.ndarray:
    """在原始值域上做 gamma 变换，参考上限取本数组自身的 max（避免写死 4095）。"""
    hi = float(vol.max())
    if hi <= 0:
        return vol
    return (np.power(np.clip(vol / hi, 0.0, 1.0), g) * hi).astype(np.float32)


def _resolve_size(files: list[Path], size: int | None) -> int:
    """确定 `self.size`（**只用于旋转/缩放的旋转中心**）。

    🔴 为什么要从数据里推断，而不是信任一个默认 512：
        `_augment` 里仿射的旋转中心写的是 `(self.size-1)/2`。如果数据其实是 768
        而 `self.size` 还是默认的 512，旋转就会绕 (255.5,255.5) 而不是 (383.5,383.5)。
        后果**不是标签错位**（seq/minip/label 三个共用同一个 mat/off，图文仍然一致），
        而是**多出一个随机的额外平移**：偏移量 = (I - R/scale)·Δc，Δc=(-128,-128)，
        768 图上最大 ≈54 px（ang=15°、scale=0.9 时）。
        ⚠️ `_augment` 第 3 步的仿射**无条件执行**（没有概率门），所以这是 **100% 训练样本**
        都会中的静默错误 —— 且 loss 曲线完全正常，看不出来。
        所以这里直接以数据的真实形状为准，并在不相符时**报错**而不是默默用错中心。

    行为：
        size=None（推荐）-> 自动取数据形状，并检查全 split 是否一致；
        size=<int>        -> 与数据不符时**直接报错**，而不是默默用错的中心。
    """
    shapes = {}
    for p in files:
        h, w = np.load(p)["label"].shape[:2]
        shapes.setdefault((int(h), int(w)), []).append(p.name)

    if len(shapes) != 1:
        detail = "；".join(f"{k}: {len(v)} 个（如 {v[0]}）" for k, v in shapes.items())
        raise ValueError(
            f"同一个 split 里出现了多种尺寸，不能混训：{detail}\n"
            f"-> 说明 processed 里有文件是不同 --size 生成的，请重新生成数据。")
    actual = next(iter(shapes))
    if actual[0] != actual[1]:
        raise ValueError(f"数据不是正方形 {actual}，fit_to_square 本应输出正方形，请检查数据。")

    if size is None:
        return actual[0]
    if int(size) != actual[0]:
        raise ValueError(
            f"传入 size={size}，但 {files[0].parent.name} 下的数据实际是 {actual[0]}"
            f"（例：{files[0].name}）。\n"
            f"-> 旋转增强的旋转中心会算错（不报错的静默错误）。"
            f"请改用 size=None 自动推断，或传 size={actual[0]}。")
    return actual[0]


class DSCADataset(Dataset):
    """读取预处理好的 .npz，返回 (seq, minip, label)。

    seq   : (T, 1, H, W) float32，已归一化
    minip : (1, H, W)    float32，已归一化
    label : (H, W)       int64，取值 0/1/2
    """

    def __init__(self, root: str | Path, split: str = "train",
                 cores: list[str] | None = None, size: int | None = None,
                 num_frames: int = 8, augment: bool = False,
                 elastic: bool = False, seed: int = 0,
                 norm: str = DEFAULT_NORM) -> None:
        self.root = Path(root) / split
        self.num_frames = num_frames
        self.augment = augment
        self.elastic = elastic
        if norm not in NORM_MODES:
            raise ValueError(f"未知归一化模式 {norm!r}，可选 {NORM_MODES}")
        self.norm = norm

        if cores is None:
            self.files = sorted(self.root.glob("*.npz"))
        else:
            self.files = [self.root / f"{c}.npz" for c in cores]
        missing = [p for p in self.files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"缺少 {len(missing)} 个文件，例如 {missing[0]}")
        if not self.files:
            raise FileNotFoundError(f"{self.root} 下没有 npz，请先运行 data/prepare.py")

        self.size = _resolve_size(self.files, size)

        self.rng = np.random.RandomState(seed)

    @property
    def cores(self) -> list[str]:
        return [p.stem for p in self.files]

    def __len__(self) -> int:
        return len(self.files)

    # ------------------------------------------------------------------
    def _augment(self, seq: np.ndarray, minip: np.ndarray,
                 label: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = self.rng

        # 1) 镜像
        if rng.random() < 0.5:
            seq, minip, label = seq[:, :, ::-1], minip[:, ::-1], label[:, ::-1]
        if rng.random() < 0.5:
            seq, minip, label = seq[:, ::-1, :], minip[::-1, :], label[::-1, :]

        # 2) 90° 旋转
        k = int(rng.randint(4))
        if k:
            seq = np.rot90(seq, k, axes=(1, 2))
            minip = np.rot90(minip, k)
            label = np.rot90(label, k)

        # 3) 旋转 + 缩放（一次仿射，三者共用）
        ang = rng.uniform(-15.0, 15.0)
        scale = rng.uniform(0.9, 1.1)
        th = np.deg2rad(ang)
        mat = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) / scale
        c = np.array([(self.size - 1) / 2.0, (self.size - 1) / 2.0])
        off = c - mat @ c
        seq = _apply_affine(seq, mat, off, order=1)
        minip = _apply_affine(minip, mat, off, order=1)
        label = _apply_affine(label.astype(np.float32), mat, off, order=0)

        # 4) 弹性形变（可选）
        if self.elastic:
            seq = _elastic(seq, alpha=20.0, sigma=6.0, rng=rng, order=1)
            minip = _elastic(minip, alpha=20.0, sigma=6.0, rng=rng, order=1)
            label = _elastic(label, alpha=20.0, sigma=6.0, rng=rng, order=0)

        # 5) gamma（只动图像，不动标签）
        #    在**原始值域**里做，用本图自身的 max 作参考 ——
        #    这样与后面选哪种归一化无关（原来写死除以 INTENSITY_DIV，
        #    会把 FXR 那批暗图压到 0~0.06，gamma 之后几乎全 0）。
        if rng.random() < 0.5:
            g = rng.uniform(0.7, 1.5)
            seq = _gamma(seq, g)
            g2 = rng.uniform(0.7, 1.5)
            minip = _gamma(minip, g2)

        label = np.clip(np.rint(label), 0, 2).astype(np.int64)
        return seq, minip, label

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        d = np.load(self.files[idx])
        seq = d["seq"].astype(np.float32)              # (T,H,W)
        minip = d["minip"].astype(np.float32)          # (H,W)
        label = d["label"].astype(np.float32)          # (H,W)

        if seq.shape[0] != self.num_frames:
            raise ValueError(f"{self.files[idx].name} 有 {seq.shape[0]} 帧，期望 {self.num_frames}")

        if self.augment:
            seq, minip, label = self._augment(seq, minip, label)
            label = label.astype(np.int64)
        else:
            label = label.astype(np.int64)

        seq = normalize(seq, self.norm).astype(np.float32)      # (T,H,W)
        minip = normalize(minip, self.norm).astype(np.float32)  # (H,W)

        return (
            torch.from_numpy(seq).unsqueeze(1),        # (T,1,H,W)
            torch.from_numpy(minip).unsqueeze(0),      # (1,H,W)
            torch.from_numpy(label),                   # (H,W)
        )
