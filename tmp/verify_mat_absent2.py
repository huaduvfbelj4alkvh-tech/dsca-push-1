"""对着源头 labelsTs 核对 8 例 MAT 缺失样本（并抽几例 MAT 正常的做正向对照）。

目的：确认 npz 里的 class 2 缺失是官方标注本来就如此，
      而不是我们的 prepare.py 在提取 / 修轴序时把 class 2 弄丢了。
"""
from __future__ import annotations

import os

import numpy as np
import nibabel as nib

RAW_LAB_TS = r"E:\Datasets\DSCA\DSA_public\labelsTs"
PROC_TEST = r"E:\Datasets\DSCA\processed\test"

ABSENT = ["GJW_AP_05", "GJW_AP_35", "GJW_L_14", "GJW_L_41",
          "LXJ_AP_07", "LXJ_AP_32", "LXJ_L_51", "LYL_AL_20"]
# 正向对照：从你贴的 per_sample 里 MAT 有分的
PRESENT = ["CXY_AP_19", "WX_AP_64", "ZQL_L_16", "ZXM_AP_20"]


def counts(arr):
    u, c = np.unique(np.asarray(arr).astype(np.int64), return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


def row(core):
    raw = os.path.join(RAW_LAB_TS, f"fu_{core}.nii.gz")
    npz = os.path.join(PROC_TEST, f"{core}.npz")
    if not os.path.exists(raw):
        return f"{core:<12} 原始 nii 不存在: {raw}"
    if not os.path.exists(npz):
        return f"{core:<12} npz 不存在: {npz}"
    a = nib.load(raw)
    raw_arr = np.asanyarray(a.dataobj)
    npz_lab = np.load(npz)["label"]
    cr, cn = counts(raw_arr), counts(npz_lab)
    same = (cr.get(1, 0) == cn.get(1, 0)) and (cr.get(2, 0) == cn.get(2, 0))
    return (f"{core:<12} raw={a.shape} {cr}\n"
            f"{'':<12} npz={npz_lab.shape} {cn}   一致={same}")


print("=" * 78)
print("A) MAT 缺失的 8 例 —— 源头是不是也没有 class 2？")
print("=" * 78)
for c in ABSENT:
    print(row(c))

print()
print("=" * 78)
print("B) 正向对照：MAT 正常的样本")
print("=" * 78)
for c in PRESENT:
    print(row(c))

print()
print("=" * 78)
print("C) 全 44 例：raw 与 npz 的 class1/class2 体素数是否逐例相同")
print("=" * 78)
n_diff, n_missing, rows = 0, 0, []
for f in sorted(os.listdir(RAW_LAB_TS)):
    core = f[3:-7]                      # 去掉 'fu_' 和 '.nii.gz'
    raw = os.path.join(RAW_LAB_TS, f)
    npz = os.path.join(PROC_TEST, core + ".npz")
    if not os.path.exists(npz):
        n_missing += 1
        continue
    cr = counts(np.asanyarray(nib.load(raw).dataobj))
    cn = counts(np.load(npz)["label"])
    if cr.get(1, 0) != cn.get(1, 0) or cr.get(2, 0) != cn.get(2, 0):
        n_diff += 1
        rows.append((core, cr, cn))

print(f"对比 {44 - n_missing} 例，缺失 npz {n_missing} 例，体素数不一致 {n_diff} 例")
for core, cr, cn in rows:
    print(f"  不一致: {core}  raw={cr}  npz={cn}")

print()
print("D) 全 44 例里，原始标注中 class 2 为空的样本数")
n_zero = 0
zero_list = []
for f in sorted(os.listdir(RAW_LAB_TS)):
    core = f[3:-7]
    cr = counts(np.asanyarray(nib.load(os.path.join(RAW_LAB_TS, f)).dataobj))
    if cr.get(2, 0) == 0:
        n_zero += 1
        zero_list.append(core)
print(f"原始标注 class2 为空: {n_zero} / 44")
print(f"名单: {zero_list}")
