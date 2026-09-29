"""轴序修复的 A/B 可视化：同一张 MinIP 上叠「旧标签」与「新标签」的轮廓。

看什么
------
每行一个样本，三列：
    1. MinIP（模型实际看到的输入）
    2. 旧 processed 的 label 轮廓（红）—— 若与血管对不上 = 转置错位
    3. 新 processed_fix 的 label 轮廓（绿）—— 应该紧贴血管
标题里给出两个版本的「标签-图像对齐 AUC」：红的那版 AUC≈0.6，绿的≈0.97。

用法
----
    E:\\Anaconda\\envs\\dsca\\python.exe tmp\\viz_axis_fix.py
输出：viz/axis_fix_compare.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.prepare import _align_auc  # noqa: E402

OLD = Path(r"E:\Datasets\DSCA\processed_axisbug_20260926")   # 旧：形状判据（有 bug）
NEW = Path(r"E:\Datasets\DSCA\processed")                    # 新：对齐度选方向
OUT = ROOT / "viz" / "axis_fix_compare.png"

# 5 个正方形（旧判据放行、被修复）+ 2 个非正方形（旧判据已救回，作对照）
SAMPLES = [
    ("GJW_AP_35", "test", "952x952 square"),
    ("CXY_AP_19", "test", "1024x1024 square"),
    ("ZQL_AP_36", "test", "512x512 square"),
    ("HXY_AL_94", "test", "512x512 square"),
    ("LXJ_L_40", "test", "952x952 square"),
    ("ZHL_AP_39", "test", "960x742 non-square (control)"),
    ("ZQL_L_16", "test", "472x512 non-square (control)"),
]


def npz(root: Path, split: str, core: str):
    p = root / split / f"{core}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["minip"].astype(np.float32), z["label"]


def main() -> int:
    fig, axes = plt.subplots(len(SAMPLES), 3, figsize=(11.5, 3.05 * len(SAMPLES)))
    if len(SAMPLES) == 1:
        axes = axes[None, :]

    for r, (core, split, note) in enumerate(SAMPLES):
        new = npz(NEW, split, core)
        old = npz(OLD, split, core)
        if new is None or old is None:
            for c in range(3):
                axes[r, c].axis("off")
                axes[r, c].text(0.5, 0.5, f"{core}: npz 缺失", ha="center", va="center")
            continue
        img, lab_new = new
        _, lab_old = old
        auc_old = _align_auc(lab_old, img)
        auc_new = _align_auc(lab_new, img)

        for c, (lab, color, tag, auc) in enumerate((
            (None, None, "MinIP (model input)", None),
            (lab_old, "red", "OLD label (shape rule passed it)", auc_old),
            (lab_new, "lime", "NEW label (alignment-selected)", auc_new),
        )):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray")
            if lab is not None:
                ax.contour((lab > 0).astype(float), levels=[0.5], colors=[color],
                           linewidths=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            title = tag if auc is None else f"{tag}\nalign AUC = {auc:.3f}"
            ax.set_title(title, fontsize=9,
                         color="black" if auc is None else
                         ("#a32d2d" if auc < 0.9 else "#0f6e56"))

        axes[r, 0].set_ylabel(f"{core}\n{note}", fontsize=9)

    fig.suptitle("DSCA label-axis fix: square samples were silently left transposed "
                 "(90 deg off)", fontsize=12)
    fig.tight_layout(rect=(0.02, 0, 1, 0.972))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"已保存 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
