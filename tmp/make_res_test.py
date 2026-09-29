"""自测 new audit_res_vs_dice.py：造两份**已知答案**的合成 eval json。
  A) 注入剂量-反应（BV 随压缩倍数下降）-> 工具应判「支持假说」
  B) 注入无关（BV 与压缩倍数独立）    -> 工具应判「不支持」
两次都要被正确判出来，才说明这个工具真的有判别力（而不是永远说"支持"）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

DATA = Path(r"E:\Datasets\DSCA\processed")
BASE = Path(r"F:\Projects\DSCA\tmp\_restest")

rep = json.loads((DATA / "report.json").read_text(encoding="utf-8"))
sizes = {}
for s in rep["splits"]["test"]["samples"]:
    w, h = s["orig"].split("x")
    sizes[s["core"]] = (int(h), int(w))

lab_absent_mat = set()
for c in sizes:
    lab = np.load(DATA / "test" / f"{c}.npz")["label"]
    if (lab == 2).sum() == 0:
        lab_absent_mat.add(c)
print("GT 无 MAT:", len(lab_absent_mat))

rng = np.random.default_rng(0)
os.makedirs(BASE, exist_ok=True)


def build(name: str, slope_bv: float, slope_mat: float, seed: int):
    r = np.random.default_rng(seed)
    per = []
    for c, (rs, cs) in sorted(sizes.items()):
        f = max(rs, cs) / 512.0
        x = np.log2(f)                       # 0 for uncompressed, ~1.0 for 2x
        bv = float(np.clip(0.86 - slope_bv * x + r.normal(0, 0.03), 0, 1))
        mat = float(np.clip(0.90 - slope_mat * x + r.normal(0, 0.03), 0, 1))
        if c in lab_absent_mat:
            mat = 8.5e-09                     # 复现"幻觉被罚 0"的真实结构
        per.append({"core": c, "background": 0.99, "BV": round(bv, 4), "MAT": mat})
    bvs = [s["BV"] for s in per]
    mats = [s["MAT"] for s in per]
    doc = {"ckpt": f"runs/{name}/best.pth", "epoch": 500, "split": "test",
           "tta": True, "fold": 0,
           "overall": {"n_samples": len(per), "per_class": {
               "background": 0.99, "BV": sum(bvs) / len(bvs),
               "MAT": sum(mats) / len(mats)},
               "mean_foreground": (sum(bvs) / len(bvs) + sum(mats) / len(mats)) / 2},
           "per_sample": per}
    d = BASE / name
    os.makedirs(d, exist_ok=True)
    (d / "eval_test.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"写出 {d/'eval_test.json'}  (注入 BV slope={slope_bv}, MAT slope={slope_mat})")


build("synth_dose_response", 0.15, 0.02, 1)   # 期望：支持
build("synth_null", 0.0, 0.0, 2)             # 期望：不支持
