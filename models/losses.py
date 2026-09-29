"""损失函数：交叉熵 + Dice，配深监督。

论文第 IV-A 节只说了损失是 CE + Dice，**没给两者的权重，也没给深监督的各项权重**。
所以这些权重都做成参数（默认 1.0 / 1.0、辅助项 0.5），方便后续消融。

------------------------------------------------------------------
Dice 的除零问题（数据实测）
------------------------------------------------------------------
    数据里存在"只有 0/1、没有 MAT(类 2)"的样本（实测 20+ 例）。
    如果按 batch 聚合各类别的交并集，缺失类别会得到 0/0。
    本实现改成**逐样本逐类别**计算，并只在"该样本中真实出现过的类别"上取平均。

------------------------------------------------------------------
🔴 Dice 要不要算背景？（2026-09-20 实测后改成默认不算）
------------------------------------------------------------------
    背景占 94~96%，一算背景，Dice 就是 0.95 起步 —— 这个数字既看不出前景好坏，
    也把前景的梯度淹掉（背景项梯度占比极高）。实测过拟合实验里
    "Dice 含背景" 与 "Dice 不含背景" 的前景 Dice 能差 0.1~0.2。
    → 默认 `include_background_in_dice=False`，想看论文口径时用 True 切回去。

------------------------------------------------------------------
CE 的类别不平衡（可选权重 / focal）
------------------------------------------------------------------
    前景只占 3~6%。纯 CE 的梯度被背景主导，模型容易停在"全预测背景"的
    局部解（此时 CE≈0.2，看着不高但前景 Dice=0）。
    → 提供 `class_weights`（按类别加权）与 `focal_gamma`（focal loss）两个旋钮。
    注意：实测显示**归一化才是主因**（跨样本 21.8 倍尺度差），
    权重/focal 只是保险，不要在没改归一化的情况下靠加大权重硬顶。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                   n_classes: int, eps: float = 1e-5,
                   include_background: bool = False) -> torch.Tensor:
    """logits: (B, C, H, W) 未归一化；target: (B, H, W) long"""
    probs = logits.softmax(dim=1)
    target_1h = F.one_hot(target, n_classes).permute(0, 3, 1, 2).to(probs.dtype)

    if not include_background:
        probs = probs[:, 1:]
        target_1h = target_1h[:, 1:]

    inter = (probs * target_1h).sum(dim=(2, 3))                       # (B, C)
    denom = probs.sum(dim=(2, 3)) + target_1h.sum(dim=(2, 3))        # (B, C)
    dice = (2 * inter + eps) / (denom + eps)                          # (B, C)

    # 只统计"该样本里真实存在的类别"，避免缺失类别用 eps/eps=1 稀释梯度
    present = target_1h.sum(dim=(2, 3)) > 0                           # (B, C)
    dice = torch.where(present, dice, torch.ones_like(dice))

    return 1.0 - dice.mean()


def inverse_frequency_weights(class_counts, n_classes: int = 3,
                              max_ratio: float = 25.0) -> torch.Tensor:
    """按类别频率算权重：w_c ∝ 1/freq_c，并做两件事防止训练被带跑偏。

    1. **限制最大/最小权重之比**（`max_ratio`），不让某一类被拉得过分激进
    2. **归一化到 `sum(w_c * freq_c) == 1`** —— 这样加权后的 CE 量级与普通 CE 一致，
       学习率不用跟着改。

    ⚠️ 这一步很关键：如果只做"让最小权重 = 1"，背景类权重是 1、前景类被抬到 25，
       CE 会从 ~1.1 直接跳到 ~20，配合 lr=0.01 的 SGD 会立刻发散。
       （实测：未归一化时 loss 从 2.00 变 21.66，差 10 倍。）

    class_counts: 长度 n_classes 的频数（像素数或比例都行，会自动归一化）
    返回: (C,) float32，满足 sum(w * freq) ≈ 1
    """
    f = torch.as_tensor(class_counts, dtype=torch.float32).clamp_min(1e-8)
    f = f / f.sum()
    w = 1.0 / f
    w = w.clamp_max(max_ratio * w.min())     # 限制最大/最小之比
    w = w * f.sum() / (w * f).sum()          # 保持损失尺度：sum(w_c * freq_c) = 1
    return w


class DeepSupervisionLoss(nn.Module):
    """主输出 + 若干深监督辅助输出的加权损失。

    辅助输出是低分辨率的，这里统一插值回标签分辨率再算损失
    （先用 logits 插值再算 CE/Dice，等价且省显存 —— 不插值标签可以避免生成
     多份 (B,H,W) 的标签副本）。

    参数量少、走的是一条 log_softmax 路径，方便 focal / 类别权重复用：
        focal_gamma = 0 且 class_weights = None  ->  普通 CE
    """

    def __init__(self, n_classes: int = 3, ce_weight: float = 1.0,
                 dice_weight: float = 1.0, aux_weight: float = 0.5,
                 include_background_in_dice: bool = False,
                 class_weights: torch.Tensor | list[float] | None = None,
                 focal_gamma: float = 0.0) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.include_background_in_dice = include_background_in_dice
        self.focal_gamma = float(focal_gamma)

        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            if w.numel() != n_classes:
                raise ValueError(f"class_weights 长度 {w.numel()} != n_classes {n_classes}")
            self.register_buffer("class_weights", w)

    # ------------------------------------------------------------------
    def _ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """(可选 focal) CE。gamma=0 且无权重时等价于 nn.CrossEntropyLoss()。"""
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)       # (B,H,W)
        if self.focal_gamma:
            pt = logpt.exp().clamp(0.0, 1.0)
            loss = -((1.0 - pt) ** self.focal_gamma) * logpt
        else:
            loss = -logpt
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(loss.dtype)[target]
        return loss.mean()

    def _single(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if self.ce_weight:
            loss = loss + self.ce_weight * self._ce(logits, target)
        if self.dice_weight:
            loss = loss + self.dice_weight * soft_dice_loss(
                logits, target, self.n_classes,
                include_background=self.include_background_in_dice,
            )
        return loss

    # ------------------------------------------------------------------
    def forward(self, main_logits: torch.Tensor, aux_logits: list[torch.Tensor],
                target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        main = self._single(main_logits, target)

        aux_term = torch.zeros((), device=main.device, dtype=main.dtype)
        if aux_logits:
            per_aux = []
            for a in aux_logits:
                if a.shape[-2:] != target.shape[-2:]:
                    a = F.interpolate(a, size=target.shape[-2:], mode="bilinear",
                                      align_corners=False)
                per_aux.append(self._single(a, target))
            aux_term = torch.stack(per_aux).mean()

        total = main + self.aux_weight * aux_term
        parts = {
            "loss_main": float(main.detach()),
            "loss_aux": float(aux_term.detach()),
            "loss_total": float(total.detach()),
        }
        return total, parts
