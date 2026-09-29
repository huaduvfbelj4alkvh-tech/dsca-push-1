"""DSANet 解码器。

论文 Fig.3 右下：解码器由若干胶囊块 + 最后的 Conv 1×1 组成。
图例给出两类块：
    (Conv 3×3 + GN + GeLU) × 2 + Up   —— 带 ×2 上采样（图里 4 个）
    (Conv 3×3 + GN + GeLU) × 2        —— 不带上采样（图里 1 个，即最右那块）
最后的灰色块是 Conv 1×1，直接输出 3 类分割图（背景 / BV / MAT）。

反推链（重要，它同时确定了编码器的深度）：
    解码器有 4 个带 Up 的块 = 4 次 ×2 上采样
    输出要回到 512×512  ->  瓶颈特征图必须是 32×32
    编码器 4 次下采样  ->  与"前 4 层带 skip"（论文正文）完全自洽

跳连来源：Fig.3 底部虚线来自 SEB 的 4 层输出。用 SEB 而不是 TEB 的理由是
SEB 每层已经注入了 TEB 的时序信息（见 encoder.py 的说明），是天然的融合特征。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .blocks import ConvBlock, Up
from .encoder import CHS


class Decoder(nn.Module):
    """融合特征 (B, 2c, 32, 32) + SEB 的 4 个 skip -> 分割 logits。

    返回 (主输出, 深监督辅助输出列表)。辅助输出是低分辨率的，训练时再插值对齐。
    """

    def __init__(self, fused_ch: int = 1024, enc_chs: tuple[int, ...] = CHS,
                 dec_chs: tuple[int, ...] = (256, 128, 64, 32),
                 n_classes: int = 3, deep_supervision: bool = True,
                 grad_checkpoint: bool = False) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision
        self.grad_checkpoint = grad_checkpoint

        # 图里那个不带 Up 的块（先压通道，再逐级上采样）
        self.stem = ConvBlock(fused_ch, dec_chs[0])

        skip_chs = tuple(enc_chs[:-1])[::-1]          # (256, 128, 64, 32)，从深到浅
        if len(skip_chs) != len(dec_chs):
            raise ValueError(f"skip 层数({len(skip_chs)}) 必须等于解码级数({len(dec_chs)})")

        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        c_in = dec_chs[0]
        for skip_c, out_c in zip(skip_chs, dec_chs):
            self.ups.append(Up(c_in, out_c))
            self.blocks.append(ConvBlock(out_c + skip_c, out_c))
            c_in = out_c

        self.head = nn.Conv2d(dec_chs[-1], n_classes, kernel_size=1, bias=True)

        # 深监督：在中间几级也接 1×1 头（论文用 CE + Dice 的深监督）
        self.aux_heads = nn.ModuleList()
        if deep_supervision:
            for out_c in dec_chs[:-1]:
                self.aux_heads.append(nn.Conv2d(out_c, n_classes, kernel_size=1, bias=True))

    def forward(self, fused: torch.Tensor,
                skips: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """fused: (B, fused_ch, h, w)；skips: 从浅到深的 4 个张量"""
        if len(skips) != len(self.ups):
            raise ValueError(f"需要 {len(self.ups)} 个 skip，收到 {len(skips)}")

        x = self.stem(fused)
        aux: list[torch.Tensor] = []
        for i, (up, block) in enumerate(zip(self.ups, self.blocks)):
            skip = skips[-1 - i]                      # 由深到浅取
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            merged = torch.cat([x, skip], dim=1)
            if self.grad_checkpoint and self.training and merged.requires_grad:
                x = cp.checkpoint(block, merged, use_reentrant=False)
            else:
                x = block(merged)
            if self.deep_supervision and i < len(self.aux_heads):
                aux.append(self.aux_heads[i](x))

        return self.head(x), aux
