"""TemporalFormer (TF) —— DSCA 论文 (TMI 2025) 时序建模模块。

论文标题:
    DSCA: A Digital Subtraction Angiography Sequence Dataset and
    Spatio-Temporal Model for Cerebral Artery Segmentation

对应原文 Fig.4 与其 caption、Eq.(1)(2)。

!! 这是"按论文从零实现"，不是官方代码的移植 !!
官方仓库 jiongzhang-john/DSCA 只有 README.md + 2 张 png，没有任何 .py。

------------------------------------------------------------------
论文明确给出的维度链（Fig.4 caption）
------------------------------------------------------------------
    (BT), c, h, w   --reshape-->   B, T, c, h, w
                    --reshape-->   (Bhw), T, c
                    --+ learnable positional embedding f_p in R^(T x c)-->
                    [residual branch: reshape -> B, (hwT), c  记作 F_s']
                    --temporal MHSA--> B, (hwT), c
                    --(+) residual--> Add & Norm
                    --rearrange--> (BT), (hw), c
                    --spatial MHSA--> B, (hwT), c
                    --(+) residual--> Add & Norm
                    --MLP (2 linear)--> --(+) residual--> Add & Norm
                    (以上重复 num_layers = 4 次)
                    --reshape--> B, T, c, h, w
                    --temporal downsample--> F_s in R^(B x c x h_hat x w_hat)

    Eq.(1): q_s, k_s, v_s = W(F_s)
    Eq.(2): F~_s = Softmax(q_s k_s^T / sqrt(d_k)) v_s

------------------------------------------------------------------
论文没有交代、由本实现**假设**的部分（每项都做成可配置开关，便于消融）
------------------------------------------------------------------
    A1 注意力头数        : 默认 num_heads = 8   (head_dim = 512 / 8 = 64)
    A2 q/k/v 投影层数     : 默认单层 Linear(dim, 3*dim)  —— 原文写作 "W(F_s)"，未提层数
    A3 MLP 隐藏维         : 默认 4 * dim = 2048
    A4 LayerNorm 位置     : 本模块自身默认 pre-LN (norm_first=True)，
                           **但 DSANet 总装时显式传 norm_first=False（post-LN）** ——
                           因为原文 Fig.3 画的是"每个子层之后都有 Add&Norm"。
                           ⚠️ 两者训练难度不同，这个开关值得消融。
    A5 位置编码加在哪几层  : 默认只在第 1 层入口加一次
    A6 "T 维下采样" 怎么做 : 默认 'maxavg'（沿 T 做 max+avg 再加），
                           **这正是 Fig.3 里 TF 输出向下接的那个 Max-Avg pool 框**。
                           其余 'conv3d' / 'conv3d_half' / 'mean' / 'last' 留作消融对照。
    A7 瓶颈 h, w          : 由输入决定。本实现骨干做 **4 次下采样**：512 -> 32，即 hw = 1024
                           （依据：解码器有 4 个带 Up 的块，且"前 4 层带 skip"，见 encoder.py）
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ----------------------------------------------------------------------------
# 基础组件
# ----------------------------------------------------------------------------
class MHSA(nn.Module):
    """多头自注意力，实现论文 Eq.(1)(2)。

    Eq.(1): q_s, k_s, v_s = W(F_s)
    Eq.(2): F~_s = Softmax(q_s k_s^T / sqrt(d_k)) v_s

    输入 (N, L, C) -> 输出 (N, L, C)。

    attn_backend ：
      'manual' —— 手工 `softmax(QK^T / sqrt(d)) V`，与论文公式一一对应，最容易读，但会把
                  完整的注意力矩阵 (N, H, L, L) 物化出来并存到反向传播。
      'sdpa'   —— 用 PyTorch 原生 F.scaled_dot_product_attention，**数学完全等价**
                  （它内部就是 softmax(QK^T/sqrt(d))V），但走 flash / memory-efficient
                  kernel，不物化注意力矩阵。
      ⚠️ 实测差距巨大（batch=1, T=8, c=512, 32x32 瓶颈，fp32）：
            manual = 3172 MiB   vs   sdpa = 1636 MiB  （省 48%）
         原因是空域 MHSA 的注意力矩阵是 (B*T, H, hw, hw) = (8,8,1024,1024) = 268 MB/张，
         4 层各存一份。**本机显存只有 5136 MiB 可用，默认必须是 'sdpa'。**
         想看公式对应关系时切回 'manual' 即可，两者数值一致性由 T3b 测试守住。
    """

    def __init__(self, dim: int, num_heads: int = 8,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 attn_backend: str = "sdpa") -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim({dim}) 必须能被 num_heads({num_heads}) 整除")
        if attn_backend not in ("manual", "sdpa"):
            raise ValueError(f"attn_backend 必须是 'manual' 或 'sdpa'，收到 {attn_backend!r}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_backend = attn_backend

        # A2: 每个 q/k/v 只做一层线性投影。用单个 Linear(dim, 3*dim) 再切分，
        #     与"三个独立 Linear"数学等价，但少一次 kernel launch。
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, length, c = x.shape
        qkv = self.qkv(x)
        # (N, L, 3, H, D) -> (3, N, H, L, D)
        qkv = qkv.reshape(n, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                     # 各 (N, H, L, D)

        if self.attn_backend == "sdpa":
            # 注意：SDPA 内部自带 1/sqrt(d_k) 缩放，不能再乘 self.scale
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            out = out.transpose(1, 2).reshape(n, length, c)
            return self.proj_drop(self.proj(out))

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (N, H, L, L)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(n, length, c)
        return self.proj_drop(self.proj(out))


class TemporalDownsample(nn.Module):
    """把 B, T, c, h, w 在 T 维压成 1，得到 F_s in R^(B x c x h x w)。

    起初论文正文只画了个 "temporal downsample" 箭头、没给算子，所以这里做了多套实现。
    后来从 Fig.3 图上读出了答案：TF 输出箭头向下接的框正是 **Max-Avg pool**（虚线框展开为
    max pool 与 avg pool 并行后相加），所以默认模式定为 'maxavg'，其余留作消融对照。

      'maxavg'      : 沿 T 做 max 与 avg 池化再加         【默认，= Fig.3 的 Max-Avg pool】
      'conv3d'      : Conv3d(kernel=(T,1,1), stride=(T,1,1))，可学习
      'conv3d_half' : 反复 Conv3d(kernel=(2,1,1), stride=(2,1,1)) 直到 T=1
      'mean'        : 沿 T 取平均
      'last'        : 只取最后一帧
    输出统一为 (B, c, h, w)。
    """

    VALID_MODES = ("maxavg", "conv3d", "conv3d_half", "mean", "last")

    def __init__(self, dim: int, num_frames: int, mode: str = "maxavg") -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode 必须是 {self.VALID_MODES} 之一，收到 {mode!r}")
        self.mode = mode
        self.num_frames = num_frames

        if mode == "conv3d":
            self.conv = nn.Conv3d(dim, dim, kernel_size=(num_frames, 1, 1),
                                  stride=(num_frames, 1, 1))
        elif mode == "conv3d_half":
            n_steps = int(math.log2(num_frames))
            if 2 ** n_steps != num_frames:
                raise ValueError(f"conv3d_half 要求 num_frames 是 2 的幂，收到 {num_frames}")
            self.convs = nn.ModuleList([
                nn.Conv3d(dim, dim, kernel_size=(2, 1, 1), stride=(2, 1, 1))
                for _ in range(n_steps)
            ])
        else:
            self.conv = None
            self.convs = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, c, h, w) -> (B, c, h, w)"""
        if self.mode == "maxavg":
            return x.amax(dim=1) + x.mean(dim=1)
        if self.mode == "mean":
            return x.mean(dim=1)
        if self.mode == "last":
            return x[:, -1]

        # Conv3d 需要在"通道在前"的布局上跑：把 T 当作深度维
        x = x.permute(0, 2, 1, 3, 4).contiguous()       # (B, c, T, h, w)
        if self.mode == "conv3d":
            x = self.conv(x)
        else:
            for conv in self.convs:
                x = conv(x)
        x = x.squeeze(2)                                # T 维已为 1
        return x


class TemporalFormerLayer(nn.Module):
    """TF 的一层：时序 MHSA -> 空域 MHSA -> MLP，每个子层 残差 + Add&Norm。

    依赖的 einsum 轴序约定（关键，T1 测试专门对拍这一点）：
      从左到右的复合轴 `(h w t)` 表示 h 最外层、w 中间、t 最内层；
      而从 (B*T, ...) 拆出的 `(b t)` 是 b 在最外层。
      两条路径在 h/w/t 上的排列必须自洽，否则时序注意力和空间注意力
      会错位——这种 bug 不会报错，只会静默降低精度。
    """

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 norm_first: bool = False, drop: float = 0.0,
                 attn_backend: str = "sdpa") -> None:
        super().__init__()
        self.norm_first = norm_first

        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = MHSA(dim, num_heads, drop, drop, attn_backend)

        self.norm_s = nn.LayerNorm(dim)
        self.attn_s = MHSA(dim, num_heads, drop, drop, attn_backend)

        self.norm_m = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, h: int, w: int, t: int,
                pos: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, h*w*t, c) -> (B, h*w*t, c)"""
        # ---------------- 1. 时序 MHSA（在 T 维做注意力，序列长度 = T）--------------
        xt = rearrange(x, 'b (h w t) c -> (b h w) t c', h=h, w=w, t=t)
        if pos is not None:
            xt = xt + pos                                # f_p in R^(T x c)，广播到 (Bhw, T, c)
        # 残差支路：论文里的 F_s'，注意位置编码在分叉之前，所以残差也含 pos
        xt_res = rearrange(xt, '(b h w) t c -> b (h w t) c', h=h, w=w, t=t)
        y = self.attn_t(self.norm_t(xt) if self.norm_first else xt)
        y = rearrange(y, '(b h w) t c -> b (h w t) c', h=h, w=w, t=t)
        x = xt_res + y
        if not self.norm_first:
            x = self.norm_t(x)

        # ---------------- 2. 空域 MHSA（在 h*w 维做注意力，序列长度 = hw）-----------
        xs = rearrange(x, 'b (h w t) c -> (b t) (h w) c', h=h, w=w, t=t)
        z = self.attn_s(self.norm_s(xs) if self.norm_first else xs)
        z = rearrange(z, '(b t) (h w) c -> b (h w t) c', h=h, w=w, t=t)
        x = x + z
        if not self.norm_first:
            x = self.norm_s(x)

        # ---------------- 3. MLP（论文：2 层线性）--------------------------------
        x = x + self.mlp(self.norm_m(x) if self.norm_first else x)
        if not self.norm_first:
            x = self.norm_m(x)
        return x


class TemporalFormer(nn.Module):
    """TemporalFormer：把 (B*T, c, h, w) 的瓶颈特征建模成 (B, c, h, w) 的时序聚合特征。

    典型用法（对应论文骨干 5 次下采样后的瓶颈）:
        tf = TemporalFormer(dim=512, num_layers=4, num_frames=8)
        bottleneck = torch.randn(2 * 8, 512, 32, 32)   # batch=2, T=8
        fused = tf(bottleneck)
        assert fused.shape == (2, 512, 32, 32)
    """

    def __init__(self, dim: int = 512, num_layers: int = 4, num_heads: int = 8,
                 mlp_ratio: float = 4.0, num_frames: int = 8,
                 norm_first: bool = False, drop: float = 0.0,
                 temporal_down: str = "maxavg",
                 pos_first_layer_only: bool = True,
                 attn_backend: str = "sdpa") -> None:
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_frames = num_frames
        self.pos_first_layer_only = pos_first_layer_only

        # A5: 可学习位置编码 f_p in R^(T x c)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList([
            TemporalFormerLayer(dim, num_heads, mlp_ratio, norm_first, drop, attn_backend)
            for _ in range(num_layers)
        ])

        self.temporal_down = TemporalDownsample(dim, num_frames, temporal_down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*T, c, h, w) -> (B, c, h, w)"""
        bt, c, h, w = x.shape
        if bt % self.num_frames != 0:
            raise ValueError(
                f"输入的 batch({bt}) 必须是 num_frames({self.num_frames}) 的整数倍；"
                f"论文把 batch 与帧数合并成 (B*T, c, h, w) 这一维。"
            )
        b = bt // self.num_frames
        t = self.num_frames

        # (B*T, c, h, w) -> B, (h*w*t), c     （直接跳到论文的 (Bhw),T,c 之后）
        x = rearrange(x, '(b t) c h w -> b (h w t) c', b=b, t=t)

        for i, layer in enumerate(self.layers):
            use_pos = self.pos_embed if (i == 0 or not self.pos_first_layer_only) else None
            x = layer(x, h=h, w=w, t=t, pos=use_pos)

        # B, (h*w*t), c -> (B, T, c, h, w) -> (B, c, h, w)
        x = rearrange(x, 'b (h w t) c -> b t c h w', h=h, w=w, t=t)
        return self.temporal_down(x)
