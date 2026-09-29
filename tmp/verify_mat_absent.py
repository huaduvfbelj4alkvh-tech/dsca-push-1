"""核对：processed/test 里 MAT 缺失的 8 例，是数据本来的性质，还是我们预处理弄丢了 class 2？

对着三处交叉验证：
  1) processed/test/*.npz            （当前线上用的，已修轴序）
  2) processed_axisbug_20260926/test （修轴序之前的，作为对照）
  3) DSA_public 里的原始 label nii.gz （源头）
"""
from __future__ import annotations

import os
from glob import glob

import numpy as np

PROC = r"E:\Datasets\DSCA\processed"
OLD = r"E:\Datasets\DSCA\processed_axisbug_20260926"
RAW = r"E:\Datasets\DSCA\DSA_public"

SUSPECT = ["GJW_AP_05", "GJW_AP_35", "GJW_L_14", "GJW_L_41",
           "LXJ_AP_07", "LXJ_AP_32", "LXJ_L_51", "LYL_AL_20"]


def npz_counts(path):
    if not os.path.exists(path):
        return None
    lab = np.load(path)["label"]
    u, c = np.unique(lab, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


print("=" * 78)
print("1) npz 标签类别统计")
print("=" * 78)
print(f"{'core':<12} {'processed(新)':<34} {'processed(轴序bug)':<34}")
for core in SUSPECT:
    a = npz_counts(os.path.join(PROC, "test", core + ".npz"))
    b = npz_counts(os.path.join(OLD, "test", core + ".npz"))
    print(f"{core:<12} {str(a):<34} {str(b):<34}")

print()
print("=" * 78)
print("2) 全测试集：MAT / BV 缺失统计")
print("=" * 78)


def scan(root):
    files = sorted(glob(os.path.join(root, "test", "*.npz")))
    if not files:
        return None
    no_mat, no_bv, tot = [], [], 0
    for f in files:
        lab = np.load(f)["label"]
        tot += 1
        if (lab == 2).sum() == 0:
            no_mat.append(os.path.basename(f)[:-4])
        if (lab == 1).sum() == 0:
            no_bv.append(os.path.basename(f)[:-4])
    return tot, no_mat, no_bv


for tag, root in (("processed(新)", PROC), ("processed(轴序bug)", OLD)):
    r = scan(root)
    if r is None:
        print(f"{tag}: 目录不存在或没有 npz")
        continue
    tot, no_mat, no_bv = r
    print(f"{tag}: 共 {tot} 例 | 无 MAT {len(no_mat)} 例 | 无 BV {len(no_bv)} 例")
    print(f"    无 MAT 名单: {no_mat}")

print()
print("=" * 78)
print("3) 源头原始 label nii.gz")
print("=" * 78)
print("DSA_public 存在:", os.path.isdir(RAW))
if os.path.isdir(RAW):
    for n in sorted(os.listdir(RAW)):
        print("   ", n)
    # 找到 label 目录
    cands = []
    for dirpath, _dirnames, filenames in os.walk(RAW):
        if any(f.endswith((".nii", ".nii.gz")) for f in filenames):
            cands.append(dirpath)
    print("  含 nii 的目录:", cands)
    if cands:
        try:
            import nibabel as nib
        except ImportError:
            print("  [WARN] 没有 nibabel，跳过原始 nii 统计")
        else:
            for core in SUSPECT:
                hits = glob(os.path.join(cands[0], f"*{core}*.nii.gz"))
                if not hits:
                    print(f"  {core:<12} 找不到原始 nii")
                    continue
                arr = np.asanyarray(nib.load(hits[0]).dataobj).astype(np.int64)
                u, c = np.unique(arr, return_counts=True)
                print(f"  {core:<12} {os.path.basename(hits[0]):<28} "
                      f"{ {int(k): int(v) for k, v in zip(u, c)} }")
