"""DSCA (DSANet) 复现 —— 模型组件。

⚠️ 官方仓库 jiongzhang-john/DSCA 只有 README.md 与两张图，没有任何 .py。
   本包是按论文 (TMI 2025) **从零实现**，不是官方代码的移植。
   汇报/简历请写"基于论文实现"，不要写"复现开源代码"。
"""

from .blocks import ConvBlock, MaxAvgPoolT, Up, pick_groups
from .decoder import Decoder
from .dsanet import DSANet, MinIPOnly
from .encoder import CHS, EncoderBranch, SpatialEncodingBranch, TemporalEncodingBranch
from .losses import DeepSupervisionLoss, inverse_frequency_weights, soft_dice_loss
from .stf import SpatialTemporalFusion
from .temporalformer import MHSA, TemporalDownsample, TemporalFormer, TemporalFormerLayer

__all__ = [
    "ConvBlock", "MaxAvgPoolT", "Up", "pick_groups",
    "Decoder",
    "DSANet", "MinIPOnly",
    "CHS", "EncoderBranch", "SpatialEncodingBranch", "TemporalEncodingBranch",
    "DeepSupervisionLoss", "soft_dice_loss", "inverse_frequency_weights",
    "SpatialTemporalFusion",
    "MHSA", "TemporalDownsample", "TemporalFormer", "TemporalFormerLayer",
]
