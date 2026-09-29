"""给 eval_calibrate.py 做端到端自测：用真实的 44 例标签 + 伪造的 per_sample。

伪造值刻意复现 MinIP-only f0 的真实结构：
  · 8 例 GT 无 MAT 的样本，给 MAT = 8.49e-09（幻觉被罚 0）
  · 其余 36 例给 MAT = 0.9017 附近
  · BV 全体 0.7926
预期：
  union MAT ≈ 0.7378，label MAT ≈ 0.9017，且 **union_MAT*44 == label_MAT*36**
  （两种口径分子相同，只差分母 —— 这是本次结论的核心恒等式）
"""
from __future__ import annotations

import json
import os
from glob import glob

import numpy as np

PROC_TEST = r"E:\Datasets\DSCA\processed\test"
OUT = r"F:\Projects\DSCA\tmp\_caltest\runs\fake_miniponly"

os.makedirs(OUT, exist_ok=True)

cores = sorted(os.path.basename(p)[:-4] for p in glob(os.path.join(PROC_TEST, "*.npz")))
assert len(cores) == 44, len(cores)

MAT_ABSENT = set()
for c in cores:
    lab = np.load(os.path.join(PROC_TEST, c + ".npz"))["label"]
    if (lab == 2).sum() == 0:
        MAT_ABSENT.add(c)
print("GT 无 MAT 的样本:", len(MAT_ABSENT))

MAT_OK = 0.9017
per_sample = []
for c in cores:
    mat = 8.49e-09 if c in MAT_ABSENT else MAT_OK
    per_sample.append({"core": c, "background": 0.9925, "BV": 0.7926, "MAT": mat})

bv_all = [s["BV"] for s in per_sample]
mat_all = [s["MAT"] for s in per_sample]
mat_ok = [s["MAT"] for s in per_sample if s["core"] not in MAT_ABSENT]
overall = {
    "n_samples": 44,
    "per_class": {"background": 0.9925, "BV": sum(bv_all) / len(bv_all),
                  "MAT": sum(mat_all) / len(mat_all)},
    "mean_foreground": (sum(bv_all) / len(bv_all) + sum(mat_all) / len(mat_all)) / 2,
    "mean_all_classes": None,
}
doc = {"ckpt": os.path.join(OUT, "best.pth"), "epoch": 130, "split": "test",
       "tta": True, "fold": 0, "overall": overall, "per_sample": per_sample}

p = os.path.join(OUT, "eval_test.json")
with open(p, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, ensure_ascii=False, indent=2)
print("写出:", p)
print("伪造 union MAT =", round(overall["per_class"]["MAT"], 6))
print("伪造 union 两类均 =", round(overall["mean_foreground"], 6))
print("手工预期 label MAT =", round(sum(mat_ok) / len(mat_ok), 6),
      f"(n={len(mat_ok)})")
print("恒等式检查: union*44 =", round(overall["per_class"]["MAT"] * 44, 6),
      "| label*36 =", round(sum(mat_ok) / len(mat_ok) * len(mat_ok), 6))
