# -*- coding: utf-8 -*-
"""Phase 0.1 分辨率审计（只读，不改任何数据）。

验证改进计划的疑点 A：
    prepare.py 把原图「长边缩放到 512 + 中心 pad」
    论文则是「从原图随机裁剪 512x512」（保留 1:1 分辨率）
若原始 DICOM 是 1024x1024，缩放到 512 会让线性分辨率减半，细血管（1-2px）几乎不可分。

输出三节：
  1) 原始 DICOM (Rows x Columns) 分布，并按 AP / 侧位拆开
  2) processed npz 的 keys / shapes / dtype
  3) 细结构损失：label 在原始分辨率 vs 缩放到 512 后的「1px 腐蚀残留率」对比
     —— 细血管会被腐蚀掉，残留率越高说明细结构越少（被融合/抹平）
"""
from pathlib import Path
from collections import Counter

import numpy as np

DSA = Path(r"E:\Datasets\DSCA\DSA_public")
PROC = Path(r"E:\Datasets\DSCA\processed")

L = []


def log(s=""):
    L.append(str(s))


log("=" * 72)
log("Phase 0.1 分辨率审计")
log("=" * 72)

# ---------------------------------------------------------------- 1) DICOM 尺寸
log("")
log("[1] 原始 DICOM (Rows x Columns) 分布")
try:
    import pydicom

    recs = []
    for sub in ("imagesTr", "imagesTs"):
        d = DSA / sub
        if not d.is_dir():
            continue
        fs = sorted(d.glob("*.dcm"))
        log("  %-9s %4d 个 .dcm" % (sub, len(fs)))
        for p in fs:
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                recs.append((sub, int(ds.Rows), int(ds.Columns), p.name))
            except Exception as e:
                log("    ! 读取失败 %s: %s" % (p.name, e))

    if recs:
        log("")
        log("  全部尺寸分布:")
        for (h, w), n in Counter((r[1], r[2]) for r in recs).most_common():
            ex = next(r[3] for r in recs if (r[1], r[2]) == (h, w))
            log("    %5d x %-5d : %4d 例   例如 %s" % (h, w, n, ex))

        log("")
        log("  按体位拆开:")
        for tag in ("_AP_", "_L_", "_AL_", "_BL_"):
            sub_r = [r for r in recs if tag in r[3]]
            if not sub_r:
                continue
            c2 = Counter((r[1], r[2]) for r in sub_r)
            s = ", ".join("%dx%d:%d" % (k[0], k[1], v) for k, v in c2.most_common())
            log("    %-6s (共 %3d 例) -> %s" % (tag, len(sub_r), s))

        big = [r for r in recs if max(r[1], r[2]) > 512]
        log("")
        log("  >>> 长边 > 512 的样本: %d / %d (%.1f%%)"
            % (len(big), len(recs), 100.0 * len(big) / max(1, len(recs))))
        log("  >>> 判定：这些样本缩放到 512 时线性分辨率被压缩 "
            "%.2f 倍（细血管同步变细）" % (np.mean([max(r[1], r[2]) / 512.0 for r in big])
                                          if big else 1.0))
except ImportError:
    log("  ! 未安装 pydicom，跳过")

# ---------------------------------------------------------------- 2) processed npz
log("")
log("[2] processed npz 结构与形状")
if PROC.is_dir():
    npzs = sorted(PROC.rglob("*.npz"))
    log("  共 %d 个 npz" % len(npzs))
    for p in npzs[:2]:
        try:
            z = np.load(p)
            log("    样本文件: %s" % p.relative_to(PROC))
            for k in z.files:
                a = z[k]
                log("        %-10s shape=%-22s dtype=%-9s min=%.4g max=%.4g"
                    % (k, str(a.shape), str(a.dtype), float(a.min()), float(a.max())))
        except Exception as e:
            log("    ! %s: %s" % (p.name, e))

    shp = Counter()
    for p in npzs:
        try:
            z = np.load(p)
            k = "seq" if "seq" in z.files else z.files[0]
            shp[(k, tuple(z[k].shape))] += 1
        except Exception:
            pass
    log("")
    log("  主数组形状分布（前 12）:")
    for (k, s), n in shp.most_common(12):
        log("    %-10s %-22s : %d 个" % (k, str(s), n))
else:
    log("  ! processed 目录不存在")

# ---------------------------------------------------------------- 3) 细结构损失
log("")
log("[3] 细结构损失：原始 label vs 缩放到 512 后（1px 腐蚀残留率）")
log("    残留率 = 腐蚀1px后前景像素 / 原前景像素；细血管会被腐蚀掉，")
log("    残留率越高 = 细结构越少。缩放若让残留率明显升高 = 细血管被抹平。")


def survival(mask):
    from scipy.ndimage import binary_erosion

    fg = mask > 0
    n = int(fg.sum())
    if n < 10:
        return None
    er = int(binary_erosion(fg, iterations=1).sum())
    return n, er, er / float(n)


try:
    import nibabel as nib
    from skimage.transform import resize as sk_resize

    labs = sorted((DSA / "labelsTr").glob("*.nii.gz"))
    log("")
    log("  labelsTr 共 %d 个 .nii.gz" % len(labs))
    for lab in labs[:8]:
        core = lab.name.replace(".nii.gz", "")
        try:
            arr = np.asarray(nib.load(str(lab)).dataobj)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            r0 = survival(arr)
            h, w = arr.shape[:2]
            sc = 512.0 / max(h, w)
            nh, nw = max(1, int(round(h * sc))), max(1, int(round(w * sc)))
            small = sk_resize(arr.astype(np.float32), (nh, nw), order=0,
                              preserve_range=True, anti_aliasing=False)
            r1 = survival(small)
            if r0 and r1:
                log("    %-24s 原始 %4dx%-4d fg=%7d 残留%.4f  ->  512域 %4dx%-4d fg=%7d 残留%.4f  %s"
                    % (core, h, w, r0[0], r0[2], nh, nw, r1[0], r1[2],
                       "细结构变少" if r1[2] > r0[2] * 1.15 else
                       ("细结构变多" if r1[2] < r0[2] * 0.85 else "基本不变")))
        except Exception as e:
            log("    ! %s: %s" % (core, e))
except Exception as e:
    log("  ! 依赖缺失: %s" % e)

out = Path(r"F:\Projects\DSCA\tmp\audit_resolution.txt")
out.write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print("\n[written] %s" % out)
