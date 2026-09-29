"""DSANet 总装 —— 论文 Fig.3 的完整网络。

整体数据流：

    DICOM 序列 (B,T,1,H,W) ─┬─> TEB ─> (B*T,512,32,32) ─> TF ─> Max-Avg pool ─> F_s ─┐
                            │                                                          ├─> STF ─> Decoder ─> 3 类
    MinIP    (B,1,H,W)  ────┴─> SEB ─> (B,512,32,32) = F_m ────────────────────────────┘
                                  ↑
                    TEB 各层经 Max-Avg pool 沿 T 聚合后逐层注入（Fig.3 的虚线，正文没写）

------------------------------------------------------------------
训练可行性（实测，决定 train.py 必须怎么写）
------------------------------------------------------------------
论文设置 batch=2 @ 512×512 在本机 RTX 3060 Laptop（可用 5136 MiB）上跑不动。
实测端到端 batch=1：

    全网络 fp32            峰值 6155 MiB   ❌
    全网络 AMP(bf16)       峰值 5192 MiB   ❌ 还差 56 MiB
    + 梯度检查点           见 tests/test_dsanet.py 的输出

所以 train.py 必须同时用：AMP + 梯度检查点 + batch=1 + 梯度累积 2 步
（梯度累积 2 步是为了数学上等效论文的 batch=2，不是省显存的手段）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import MaxAvgPoolT
from .decoder import Decoder
from .encoder import CHS, SpatialEncodingBranch, TemporalEncodingBranch
from .stf import SpatialTemporalFusion
from .temporalformer import TemporalFormer


class DSANet(nn.Module):
    """DSCA 论文提出的 Spatio-Temporal Model。

    最小用法：
        model = DSANet()
        seq   = torch.randn(1, 8, 1, 512, 512)   # 8 帧序列
        minip = torch.randn(1, 1, 512, 512)      # 对应的 MinIP
        main, aux = model(seq, minip)
        assert main.shape == (1, 3, 512, 512)
    """

    def __init__(self,
                 in_ch: int = 1,
                 num_frames: int = 8,
                 chs: tuple[int, ...] = CHS,
                 tf_layers: int = 4,
                 tf_heads: int = 8,
                 n_classes: int = 3,
                 deep_supervision: bool = True,
                 alpha_mode: str = "attention",
                 share_qkv: bool = False,
                 attn_backend: str = "sdpa",
                 norm_first: bool = False,
                 temporal_down: str = "maxavg",
                 use_injection: bool = True,
                 pool_first_layer: bool = False,
                 grad_checkpoint: bool = False) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.use_injection = use_injection
        self.chs = tuple(chs)
        self.grad_checkpoint = grad_checkpoint

        self.teb = TemporalEncodingBranch(in_ch, chs, pool_first_layer, grad_checkpoint)
        self.seb = SpatialEncodingBranch(in_ch, chs, pool_first_layer, grad_checkpoint)
        self.maxavg = MaxAvgPoolT()

        self.tf = TemporalFormer(
            dim=chs[-1], num_layers=tf_layers, num_heads=tf_heads,
            num_frames=num_frames, temporal_down=temporal_down,
            norm_first=norm_first, attn_backend=attn_backend,
        )
        self.stf = SpatialTemporalFusion(
            dim=chs[-1], alpha_mode=alpha_mode, share_qkv=share_qkv,
        )
        self.decoder = Decoder(
            fused_ch=2 * chs[-1], enc_chs=chs, n_classes=n_classes,
            deep_supervision=deep_supervision, grad_checkpoint=grad_checkpoint,
        )

    # ------------------------------------------------------------------
    def forward(self, seq: torch.Tensor, minip: torch.Tensor
                ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """seq: (B, T, 1, H, W)；minip: (B, 1, H, W)

        返回 (主输出 logits (B,C,H,W), 深监督辅助输出列表)。
        """
        if seq.dim() != 5:
            raise ValueError(f"seq 需要 5 维 (B,T,C,H,W)，收到 {tuple(seq.shape)}")
        b, t, c, h, w = seq.shape
        if t != self.num_frames:
            raise ValueError(f"帧数应为 {self.num_frames}，收到 {t}")
        if minip.shape != (b, c, h, w):
            raise ValueError(
                f"minip 形状应为 {(b, c, h, w)}，收到 {tuple(minip.shape)}"
            )

        seq_in = seq.reshape(b * t, c, h, w)
        if self.grad_checkpoint and self.training:
            # 梯度检查点要求输入参与计算图，否则该 stage 的重算不会建立梯度
            seq_in = seq_in.requires_grad_(True)
            minip = minip.requires_grad_(True)

        # ---- 1. TEB：论文把 B 与 T 合并后喂进去 ----
        feat_t, skips_t = self.teb(seq_in)

        # ---- 2. TEB 各层沿 T 做 Max-Avg pool -> 逐层注入 SEB ----
        injections: list[torch.Tensor] | None = None
        if self.use_injection:
            injections = []
            for f in (*skips_t, feat_t):                # 4 个 skip + 1 个瓶颈
                _, cc, hh, ww = f.shape
                injections.append(self.maxavg(f.reshape(b, t, cc, hh, ww)))

        # ---- 3. SEB：空间分支，逐层接收时序信息 ----
        feat_m, skips_m = self.seb(minip, injections=injections)

        # ---- 4. TF：建模帧间/帧内关系，出口就是 Max-Avg pool（T 维聚合）----
        f_s = self.tf(feat_t)                           # (B, 512, 32, 32)

        # ---- 5. STF：时空融合 ----
        fused = self.stf(feat_m, f_s)                   # (B, 1024, 32, 32)

        # ---- 6. 解码器 ----
        return self.decoder(fused, skips_m)


class MinIPOnly(nn.Module):
    """论文 Table VI 的 MinIP-only 基线：砍掉 TEB / TF / STF，只留 SEB + Decoder。

    用途：判断「与论文的差距」出在基础配方还是时序模块。
    forward 签名与 DSANet 一致，所以 Dataset / 训练循环 / 评估脚本都不用改。
    """

    def __init__(self,
                 in_ch: int = 1,
                 chs: tuple[int, ...] = CHS,
                 n_classes: int = 3,
                 deep_supervision: bool = True,
                 pool_first_layer: bool = False,
                 grad_checkpoint: bool = False) -> None:
        super().__init__()
        self.seb = SpatialEncodingBranch(in_ch, chs, pool_first_layer, grad_checkpoint)
        self.decoder = Decoder(
            fused_ch=chs[-1],          # 512，不是 1024：没有 STF 做 concat
            enc_chs=chs, n_classes=n_classes,
            deep_supervision=deep_supervision, grad_checkpoint=grad_checkpoint,
        )

    def forward(self, seq: torch.Tensor, minip: torch.Tensor
                ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # seq 收到但不用 —— 保持签名一致，训练 / 评估代码零改动
        feat_m, skips_m = self.seb(minip, injections=None)
        return self.decoder(feat_m, skips_m)
