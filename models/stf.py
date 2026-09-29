"""STF: Spatio-Temporal Fusion —— 论文 Fig.3 右下绿框，Eq.(3)(4)。

------------------------------------------------------------------
论文明确给出的公式
------------------------------------------------------------------
    Eq.(3):
        α_i = Softmax(Q_m K_m^T / sqrt(c))
        α_s = Softmax(Q_s K_s^T / sqrt(c))
        F~'_m = (α_i + α_s) V_m
        F~'_s = (α_i + α_s) V_s
    Eq.(4):
        F~'_sm = Concat(F~'_m, F~'_s)
        F̂_sm  = UPSAMPLE(F~'_sm) + Concat(F_s, F_m)

其中 "+" 号（α_i + α_s）是论文正文点明的"共享门控"：两个模态的注意力图相加后
**同一个门控同时作用于两条支路**，这是 STF 与普通 cross-attention 的关键区别。

从 Fig.3 图里读到、正文没写的部分：
    - "Maxpool 两路"：P（Maxpool）作用在 Q/K/V 投影**之前**，把空间尺寸从 h×w 降到 h/2×w/2。
      这正好解释了正文里 α ∈ R^(B × hw/4) 中的 hw/4。
    - Q/K/V 是 (Conv 3×3 + GN + GeLU) 单层卷积块，不是 Linear。
    - 两路各自有独立的粉/黄/绿三色块（= Q/K/V），图上画了两组，故本实现默认不共享权重。
    - UPSAMPLE 是 ×2（把 h/2×w/2 恢复回 h×w），这解释了为什么 Eq.(4) 能直接相加。

------------------------------------------------------------------
论文自相矛盾处（复现最大的坑，已做成开关）
------------------------------------------------------------------
正文写 α ∈ R^(B × hw/4)，即"每个空间位置一个标量"；
但按字面公式 Softmax(Q K^T)，矩阵乘法得到的应当是 α ∈ R^(B × hw/4 × hw/4)。
两种读法形状都能自洽，行为完全不同：

    'attention'    : α = Softmax(QK^T/√c) —— 一张 (N,N) 的空间注意力图，
                     F~' = α V。这是标准注意力读法。              【默认】
    'spatial_gate' : α = Softmax_N(Σ_c Q⊙K / √c) —— 每个位置一个标量，
                     F~' = α ⊙ V（广播乘）。这是"空间显著性门控"读法。

两者参数量完全相同，只能靠**论文的消融数字**判哪个对。所以两个都实现、开关切换。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .blocks import ConvBlock


class SpatialTemporalFusion(nn.Module):
    """STF：把空间分支特征 F_m 与时序分支特征 F_s 融合。

    典型用法（论文瓶颈：c=512，h=w=32）：
        stf = SpatialTemporalFusion(dim=512)
        f_m = torch.randn(2, 512, 32, 32)   # 来自 SEB
        f_s = torch.randn(2, 512, 32, 32)   # 来自 TF + Max-Avg pool
        out = stf(f_m, f_s)                 # (2, 1024, 32, 32)
    """

    def __init__(self, dim: int = 512, qkv_conv: int = 1,
                 share_qkv: bool = False,
                 alpha_mode: str = "attention",
                 pool_first: bool = True) -> None:
        super().__init__()
        if alpha_mode not in ("attention", "spatial_gate"):
            raise ValueError(f"alpha_mode 必须是 'attention' 或 'spatial_gate'，收到 {alpha_mode!r}")
        self.dim = dim
        self.share_qkv = share_qkv
        self.alpha_mode = alpha_mode
        self.pool_first = pool_first
        self.scale = dim ** -0.5

        # Fig.3 里的 "P"（Maxpool），作用在 Q/K/V 投影之前
        self.pool = nn.MaxPool2d(2)

        # Q/K/V 投影块：(Conv 3×3 + GN + GeLU) × 1
        if share_qkv:
            self.qkv_m = nn.ModuleList([ConvBlock(dim, dim, qkv_conv) for _ in range(3)])
            self.qkv_s = self.qkv_m
        else:
            self.qkv_m = nn.ModuleList([ConvBlock(dim, dim, qkv_conv) for _ in range(3)])
            self.qkv_s = nn.ModuleList([ConvBlock(dim, dim, qkv_conv) for _ in range(3)])

    # ------------------------------------------------------------------
    def _qkv(self, x: torch.Tensor, which: str) -> tuple[torch.Tensor, ...]:
        """x: (B, c, h, w) -> 三个 (B, N, c)，N = (h/2)*(w/2)"""
        h_c, w_c = x.shape[-2:]
        if self.pool_first:
            x = self.pool(x)
            h_c, w_c = x.shape[-2:]
        proj = self.qkv_m if which == "m" else self.qkv_s
        outs = []
        for conv in proj:
            t = conv(x)                            # (B, c, h', w')
            outs.append(rearrange(t, "b c h w -> b (h w) c"))
        return tuple(outs), (h_c, w_c)

    # ------------------------------------------------------------------
    def forward(self, f_m: torch.Tensor, f_s: torch.Tensor) -> torch.Tensor:
        """f_m, f_s: (B, c, h, w) -> (B, 2c, h, w)"""
        if f_m.shape != f_s.shape:
            raise ValueError(f"F_m 与 F_s 形状必须一致，收到 {tuple(f_m.shape)} vs {tuple(f_s.shape)}")

        (q_m, k_m, v_m), hw = self._qkv(f_m, "m")
        (q_s, k_s, v_s), _ = self._qkv(f_s, "s")
        b, n, c = q_m.shape

        if self.alpha_mode == "attention":
            # Eq.(3) 字面读法：α = Softmax(Q K^T / √c) ∈ R^(B, N, N)
            alpha_m = (q_m @ k_m.transpose(-2, -1)) * self.scale
            alpha_m = alpha_m.softmax(dim=-1)
            alpha_s = (q_s @ k_s.transpose(-2, -1)) * self.scale
            alpha_s = alpha_s.softmax(dim=-1)
            alpha = alpha_m + alpha_s                    # 共享门控 α_i + α_s
            out_m = alpha @ v_m                          # (B, N, c)
            out_s = alpha @ v_s
        else:
            # 正文维度写法的读法：每个位置一个标量门控 ∈ R^(B, N)
            sim_m = (q_m * k_m).sum(dim=-1) * self.scale          # (B, N)
            alpha_m = sim_m.softmax(dim=-1)
            sim_s = (q_s * k_s).sum(dim=-1) * self.scale
            alpha_s = sim_s.softmax(dim=-1)
            alpha = alpha_m + alpha_s                            # (B, N)
            out_m = v_m * alpha.unsqueeze(-1)
            out_s = v_s * alpha.unsqueeze(-1)

        # Eq.(3) 尾部：Concat(F~'_m, F~'_s)
        out = torch.cat([out_m, out_s], dim=-1)                  # (B, N, 2c)
        out = rearrange(out, "b (h w) c -> b c h w", h=hw[0], w=hw[1])

        # Eq.(4)：UPSAMPLE ×2 回到原分辨率
        out = F.interpolate(out, size=f_m.shape[-2:], mode="bilinear", align_corners=False)

        # Eq.(4)：+ Concat(F_s, F_m)   —— 按论文原文顺序 (F_s, F_m)
        skip = torch.cat([f_s, f_m], dim=1)                      # (B, 2c, h, w)
        return out + skip
