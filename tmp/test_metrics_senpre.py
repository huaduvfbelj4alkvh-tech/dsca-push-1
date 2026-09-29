# -*- coding: utf-8 -*-
"""`metrics.confuse_per_class / stats_per_class / aggregate_stats` 的手算校验。

为什么值得写：SEN/PRE 是我用来判定「BV 是漏检还是过检」的依据。
指标算错 -> 诊断结论反。所以每个算例的期望值都是**手算**出来的，不引用实现自身。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from metrics import aggregate_stats, stats_per_class          # noqa: E402

L: list[str] = []
PASS = FAIL = 0


def log(s=""):
    L.append(str(s))
    print(s)


def chk(tag, got, want, tol=1e-4):
    global PASS, FAIL
    if want is None:
        ok = got is None
    elif got is None:
        ok = False
    else:
        ok = abs(float(got) - float(want)) <= tol
    if ok:
        PASS += 1
        log(f"    ✅ {tag:<28} = {got}")
    else:
        FAIL += 1
        log(f"    ❌ {tag:<28} = {got}   期望 {want}")


E = 1e-5        # metrics.py 里的 eps

log("=" * 70)
log("算例 A：完全相同 -> 全 1")
gt = np.array([[0, 1], [1, 1]])
st = stats_per_class(gt.copy(), gt)
chk("A BV dice", st["BV"]["dice"], 1.0)
chk("A BV sen", st["BV"]["sen"], 1.0)
chk("A BV pre", st["BV"]["pre"], 1.0)
chk("A BV jac", st["BV"]["jac"], 1.0)
chk("A background dice", st["background"]["dice"], 1.0)
chk("A MAT present", st["MAT"]["present"], 0)          # False -> 期望 0

log("\n算例 B：GT 有 BV，预测全背景（整类漏检）")
gt = np.array([[0, 1], [1, 1]])
pr = np.zeros((2, 2), dtype=np.int64)
st = stats_per_class(pr, gt)
chk("B BV tp/fp/fn", st["BV"]["tp"], 0); chk("B BV fp", st["BV"]["fp"], 0)
chk("B BV fn", st["BV"]["fn"], 3)
chk("B BV dice≈eps/3", st["BV"]["dice"], E / (3 + E))
chk("B BV sen≈eps/3", st["BV"]["sen"], E / (3 + E))
chk("B BV pre=0(分母0)", st["BV"]["pre"], 0.0)
chk("B background dice 2/5", st["background"]["dice"], (2 * 1 + E) / (2 + 3 + E))
chk("B background pre 1/4", st["background"]["pre"], (1 + E) / (1 + 3 + E))

log("\n算例 C：GT 全背景，预测全是 BV（整类幻觉）")
gt = np.zeros((2, 2), dtype=np.int64)
pr = np.ones((2, 2), dtype=np.int64)
st = stats_per_class(pr, gt)
chk("C BV fp", st["BV"]["fp"], 4)
chk("C BV sen=None(分母0)", st["BV"]["sen"], None)
chk("C BV pre≈eps/4", st["BV"]["pre"], E / (4 + E))
chk("C background pre=0", st["background"]["pre"], 0.0)
chk("C BV present", st["BV"]["present"], 1)

log("\n算例 D：两边都没有 MAT -> present=False，全 None")
gt = np.zeros((2, 2), dtype=np.int64)
st = stats_per_class(gt.copy(), gt)
chk("D MAT present", st["MAT"]["present"], 0)
chk("D MAT dice", st["MAT"]["dice"], None)
chk("D MAT sen", st["MAT"]["sen"], None)
chk("D MAT pre", st["MAT"]["pre"], None)
chk("D background dice", st["background"]["dice"], 1.0)

log("\n算例 E：手算混合（2x3）")
# GT   [[1,1,0],
#       [0,2,2]]      PRED [[1,0,0],
#                            [0,2,0]]
gt = np.array([[1, 1, 0], [0, 2, 2]])
pr = np.array([[1, 0, 0], [0, 2, 0]])
st = stats_per_class(pr, gt)
# BV: tp=(0,0)=1, fp=0, fn=(0,1)=1
chk("E BV tp", st["BV"]["tp"], 1); chk("E BV fp", st["BV"]["fp"], 0)
chk("E BV fn", st["BV"]["fn"], 1)
chk("E BV dice 2/3", st["BV"]["dice"], (2 * 1 + E) / (2 + 0 + 1 + E))
chk("E BV sen 1/2", st["BV"]["sen"], (1 + E) / (2 + E))
chk("E BV pre 1", st["BV"]["pre"], 1.0)
chk("E BV jac 1/3", st["BV"]["jac"], (1 + E) / (1 + 0 + 1 + E))
# MAT: tp=(1,1)=1, fp=0, fn=(1,2)=1  -> 与 BV 同形
chk("E MAT dice 2/3", st["MAT"]["dice"], (2 * 1 + E) / (2 + 0 + 1 + E))
# 背景: GT==0 只有 (0,2),(1,0) 两像素；pred==0 有 (0,1),(0,2),(1,0),(1,2) 四像素
#       -> tp=2, fp=2 (两个血管像素被判成背景), fn=0  => dice = 4/6
chk("E background tp", st["background"]["tp"], 2)
chk("E background fp", st["background"]["fp"], 2)
chk("E background fn", st["background"]["fn"], 0)
chk("E background dice 4/6", st["background"]["dice"], (2 * 2 + E) / (2 * 2 + 2 + E))

log("\n算例 F：aggregate_stats —— macro(label) 与 micro 必须能分开")
# D : GT 无 MAT、预测无 MAT      -> present=False，macro 剔除
# E : GT 有 MAT(2px)，预测中 1   -> MAT dice = 2/3，计入 macro
# B2: GT 有 MAT(2px)，预测整类漏 -> MAT dice≈eps/2，**也计入 macro**（把均值拉下来）
gtD, prD = np.zeros((2, 2), np.int64), np.zeros((2, 2), np.int64)
gtE, prE = np.array([[1, 1, 0], [0, 2, 2]]), np.array([[1, 0, 0], [0, 2, 0]])
gtB, prB = np.array([[2, 2], [0, 1]]), np.zeros((2, 2), np.int64)

sD = stats_per_class(prD, gtD)
sE = stats_per_class(prE, gtE)
sB = stats_per_class(prB, gtB)
agg = aggregate_stats([sD, sE, sB])
chk("F n_samples", agg["n_samples"], 3)
chk("F D 的 MAT 是否 present", sD["MAT"]["present"], 0)
chk("F B2 的 MAT 是否 present", sB["MAT"]["present"], 1)
# macro MAT 分母 = 2（E 与 B2），D 被剔除
chk("F macro MAT n", agg["macro"]["MAT"]["n"], 2)
chk("F macro MAT = mean(2/3, eps/2)", agg["macro"]["MAT"]["dice"],
    (sE["MAT"]["dice"] + sB["MAT"]["dice"]) / 2)
# micro MAT: TP=1+0=1 ; FP=0 ; FN=1+2=3  -> dice = 2/5
chk("F micro MAT tp", agg["micro"]["MAT"]["tp"], 1)
chk("F micro MAT fn", agg["micro"]["MAT"]["fn"], 3)
chk("F micro MAT dice 2/5", agg["micro"]["MAT"]["dice"], (2 * 1 + E) / (2 * 1 + 0 + 3 + E))
# BV: D 里两边都没有 -> 也不进 macro；E 与 B2 有 -> n=2
chk("F D 的 BV 是否 present", sD["BV"]["present"], 0)
chk("F macro BV n", agg["macro"]["BV"]["n"], 2)
# 背景: 三个样本都有 -> n=3
chk("F macro background n", agg["macro"]["background"]["n"], 3)
# 两类均 = mean(macro BV dice, macro MAT dice)
chk("F macro 两类均", agg["macro"]["mean_foreground"],
    (agg["macro"]["BV"]["dice"] + agg["macro"]["MAT"]["dice"]) / 2)

log("\n" + "=" * 70)
log(f"结果：PASS {PASS}  FAIL {FAIL}")
log("=" * 70)

out = Path(__file__).with_suffix(".txt")
out.write_text("\n".join(L), encoding="utf-8")
sys.exit(1 if FAIL else 0)
