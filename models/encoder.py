"""DSANet 的双编码分支：TEB（时序）+ SEB（空间）。

------------------------------------------------------------------
论文 Fig.3 给出的结构
------------------------------------------------------------------
    TEB（Temporal Encoding Branch）
        吃整段序列。论文原文提到为训练方便把 B 和 T 合并
        [页眉 2519]，所以实际输入是 (B*T, 1, H, W)。
    SEB（Spatial Encoding Branch）
        吃 MinIP 单帧，输入 (B, 1, H, W)。

    两支结构完全相同（图例里都是同一组胶囊块），通道 32 → 64 → 128 → 256 → 512，
    但**权重不共享**（图里画了两组独立的块）。

------------------------------------------------------------------
🔴 图里有、正文没写的关键连接（本实现的重点）
------------------------------------------------------------------
Fig.3 中 TEB 每一层下方都有虚线汇入一个 "Max-Avg pool" 框，
再从该框连到 SEB 的对应层。也就是说：

    TEB 第 i 层输出 (B*T, c_i, h_i, w_i)
        → 沿 T 维做 Max-Avg pool
        → (B, c_i, h_i, w_i)
        → 注入 SEB 第 i 层输出

所以 SEB 不是纯空间分支 —— 它每一层都在接收逐层的时序信息。
这条链路在图上非常明显，但论文正文完全没提，只有读图才能发现。

------------------------------------------------------------------
层数 / 分辨率的判定依据
------------------------------------------------------------------
图例写"第一层 = Pool + (Conv3×3+GN+GeLU)×2"，但从**解码器**反推更可靠：
Fig.3 解码器有 4 个带 Up 的块，即 4 次 ×2 上采样。
512 = 32 × 2^4  ->  瓶颈特征图必须是 32×32  ->  编码器做 4 次下采样。
这也与"前 4 层带 skip 输出"（论文正文明确）自洽：4 个 skip 对 4 次上采样。
所以本实现按 **4 次下采样 -> 32×32** 实现；若日后发现应按图例做 5 次
（-> 16×16），把 pool_first_layer 打开即可。

------------------------------------------------------------------
梯度检查点（grad_checkpoint）
------------------------------------------------------------------
实测：完整 DSANet 在 batch=1 + AMP 下端到端峰值 5192 MiB，
而本机可用只有 5136 MiB —— 差 56 MiB 就装不下。
编码器前几层的 512×512 分辨率激活是绝对大头，所以对每个 stage 做
`torch.utils.checkpoint`（反向时重算前向，不保存中间激活）。
代价是训练变慢约 20~30%，换来的是能装下。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .blocks import ConvBlock

#: 五个 stage 的输出通道（论文：瓶颈 c = 512，见 Fig.4 caption）
CHS: tuple[int, ...] = (32, 64, 128, 256, 512)


class EncoderBranch(nn.Module):
    """单个编码分支：5 个 stage，stage 之间用 MaxPool2d(2) 下采样。

    输入  (B, 1, 512, 512)  ->  输出 (B, 512, 32, 32) + 4 个 skip
    """

    def __init__(self, in_ch: int = 1, chs: tuple[int, ...] = CHS,
                 pool_first_layer: bool = False,
                 grad_checkpoint: bool = False) -> None:
        super().__init__()
        self.chs = tuple(chs)
        self.pool_first_layer = pool_first_layer
        self.grad_checkpoint = grad_checkpoint
        self.pool = nn.MaxPool2d(2)

        self.stages = nn.ModuleList()
        cin = in_ch
        for c in self.chs:
            self.stages.append(ConvBlock(cin, c, n_conv=2))
            cin = c

    def forward(self, x: torch.Tensor,
                injections: list[torch.Tensor | None] | None = None
                ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """x: (B, in_ch, H, W)

        injections: 长度 = 层数 的列表，第 i 项是注入到第 i 层的 (B, c_i, h_i, w_i)；
                    None 表示该层不注入。SEB 用它接收 TEB 的逐层时序聚合。
        """
        if injections is not None and len(injections) != len(self.stages):
            raise ValueError(f"injections 长度({len(injections)}) 必须等于层数({len(self.stages)})")

        skips: list[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            if i > 0 or self.pool_first_layer:
                x = self.pool(x)
            if self.grad_checkpoint and self.training and x.requires_grad:
                # use_reentrant=False：与 AMP autocast 兼容，且不要求输入是叶子张量
                x = cp.checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
            if injections is not None and injections[i] is not None:
                inj = injections[i]
                if inj.shape != x.shape:
                    raise ValueError(
                        f"第 {i} 层注入形状 {tuple(inj.shape)} 与该层输出 {tuple(x.shape)} 不一致"
                    )
                x = x + inj                       # 逐层注入 TEB 的时序信息
            if i < len(self.stages) - 1:          # 前 4 层输出 skip（论文正文明确）
                skips.append(x)
        return x, skips


class TemporalEncodingBranch(EncoderBranch):
    """TEB：吃 (B*T, 1, H, W)。"""

    def __init__(self, in_ch: int = 1, chs: tuple[int, ...] = CHS,
                 pool_first_layer: bool = False, grad_checkpoint: bool = False) -> None:
        super().__init__(in_ch, chs, pool_first_layer, grad_checkpoint)


class SpatialEncodingBranch(EncoderBranch):
    """SEB：吃 (B, 1, H, W)，逐层接收 TEB 的时序聚合。"""

    def __init__(self, in_ch: int = 1, chs: tuple[int, ...] = CHS,
                 pool_first_layer: bool = False, grad_checkpoint: bool = False) -> None:
        super().__init__(in_ch, chs, pool_first_layer, grad_checkpoint)
