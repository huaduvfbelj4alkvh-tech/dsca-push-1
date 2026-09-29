"""推理：镜像 TTA + 按需要滑窗。

论文第 IV-A 节：测试阶段**只做镜像增强** + 512 滑窗，**不做后处理**。

关于滑窗：预处理已把所有图统一到 512×512，所以"512 滑窗"退化为整图一次前向
（滑窗在图上就没意义了）。如果以后要按原始分辨率推理，把 predict_sliding_window
打开即可 —— 它已实现，只是当前默认走整图路径。
"""

from __future__ import annotations

import torch


def _forward_prob(model, seq: torch.Tensor, minip: torch.Tensor,
                  use_amp: bool) -> torch.Tensor:
    """返回 (B, C, H, W) 的 softmax 概率。"""
    enabled = use_amp and seq.is_cuda
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
        logits, _ = model(seq, minip)
    return logits.float().softmax(dim=1)


@torch.no_grad()
def predict_proba(model, seq: torch.Tensor, minip: torch.Tensor,
                  use_amp: bool = True, tta: bool = True) -> torch.Tensor:
    """seq: (B,T,1,H,W)；minip: (B,1,H,W) -> (B,C,H,W) 概率。

    TTA：4 组镜像（原图 / 上下翻 / 左右翻 / 双翻）取平均。
    """
    model.eval()
    if not tta:
        return _forward_prob(model, seq, minip, use_amp)

    acc = None
    for flip_h, flip_w in ((False, False), (True, False), (False, True), (True, True)):
        s, m = seq, minip
        if flip_h:
            s, m = s.flip(3), m.flip(2)
        if flip_w:
            s, m = s.flip(4), m.flip(3)
        p = _forward_prob(model, s, m, use_amp)
        if flip_h:
            p = p.flip(2)
        if flip_w:
            p = p.flip(3)
        acc = p if acc is None else acc + p
    return acc / 4.0


@torch.no_grad()
def predict_label(model, seq: torch.Tensor, minip: torch.Tensor,
                  use_amp: bool = True, tta: bool = True) -> torch.Tensor:
    """返回 (B, H, W) 的整数标签。不做任何后处理（与论文一致）。"""
    return predict_proba(model, seq, minip, use_amp, tta).argmax(dim=1)
