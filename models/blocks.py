"""DSANet 的基础组件。

论文 Fig.3 图例（原图底部）给出的全部基本单元：

    Pool + (Conv 3×3 + GN + GeLU) × 2     —— 编码器第 1 层
    (Conv 3×3 + GN + GeLU) × 2            —— 编码器其余层 / 解码器块
    (Conv 3×3 + GN + GeLU) × 2 + Up       —— 解码器块（带上采样）
    Conv 3×3 + GN + GeLU                  —— STF 内 Q/K/V 投影块（单层）
    Conv 1×1                              —— 分割头
    Maxpool                               —— 下采样 / STF 内的 "P"
    Max-Avg pool                          —— 见下方 MaxAvgPoolT

【重点】Max-Avg pool 的真实含义（论文正文没写，只在 Fig.3 的虚线展开框里）：
        输入 ──┬──> max pool ──┐
               └──> avg pool ──┴──> ⊕
    即 maxpool(x) + avgpool(x)。并且它在本网络里**有两处完全不同的用法**：
      1) 沿**时间维 T**做聚合（TEB→SEB 注入、TF 输出→F_s），见 MaxAvgPoolT
      2) STF 内部沿**空间**做 2×2 下采样，那就是普通的 nn.MaxPool2d(2)
    这两处极易混淆 —— 前者的对象是 T，后者是 h/w。本文件把它们分开实现。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def pick_groups(channels: int, want: int = 8) -> int:
    """GroupNorm 要求 channels % num_groups == 0，找一个不超过 want 的合法值。

    论文图例只写了 "GN"，没给组数；nnUNet 惯例是 8 或 16。
    """
    for g in range(min(want, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    """(Conv 3×3 + GN + GeLU) × n_conv。

    n_conv=2 是编码器/解码器里那个"胶囊"块；n_conv=1 是 STF 里的 Q/K/V 投影块。
    """

    def __init__(self, cin: int, cout: int, n_conv: int = 2, norm_groups: int = 8) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c = cin
        for _ in range(n_conv):
            layers += [
                nn.Conv2d(c, cout, kernel_size=3, padding=1, bias=True),
                nn.GroupNorm(pick_groups(cout, norm_groups), cout),
                nn.GELU(),
            ]
            c = cout
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MaxAvgPoolT(nn.Module):
    """沿时间维做 max 与 avg 池化，再相加 —— 论文里的 "Max-Avg pool"。

    输入 (B, T, c, h, w) -> 输出 (B, c, h, w)。

    这是**在 T 维上的聚合**，不是空间下采样。它的作用是：
      - TEB 每层输出 → 聚合成单帧空间特征 → 注入 SEB 对应层
      - TF 输出 → 聚合成 F_s ∈ R^(B×c×h×w) → 送进 STF
    也就是说，论文说的 "temporal downsample" 实际就是这个算子
    （Fig.3 里 TF 输出向下接的就是 Max-Avg pool），并不是可学习的卷积。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"MaxAvgPoolT 需要 5 维输入 (B,T,c,h,w)，收到 {tuple(x.shape)}")
        return x.amax(dim=1) + x.mean(dim=1)


class Up(nn.Module):
    """解码器里的 "Up"：双线性上采样 ×2 + Conv 1×1 调整通道。

    论文图例只写了 "+ Up"，没给算子。用双线性 + 1×1 卷积比转置卷积更稳
    （不会产生棋盘伪影），也是 nnUNet 之外的医学分割实现里最常见的做法。
    """

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.proj = nn.Conv2d(cin, cout, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.up(x))
