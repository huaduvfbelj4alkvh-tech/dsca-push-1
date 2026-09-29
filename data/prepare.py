"""P0 数据管线：把 DSCA 原始数据转成训练可用的 .npz。

------------------------------------------------------------------
原始数据的**实测**情况（和 Zenodo 页面上的描述不一致，别信描述）
------------------------------------------------------------------
    imagesTr/*.dcm     180 个，每个是 8 帧的多帧 DICOM 序列
    labelsTr/*.nii.gz  180 个，2D 单张
    imagesTs / labelsTs 44 个
    ⚠️ 图像与标签的**文件名前缀不同**（pro_ / fu_），必须用中间的核心串配对。
    ⚠️ 数据里**没有现成的 MinIP**，Zenodo 上写的 imagesTr_s/labelsTr_s 目录并不存在。

------------------------------------------------------------------
每一步都对应一个实测踩到的坑
------------------------------------------------------------------
    1. 配对     靠核心串（去掉 pro_/fu_ 前缀与扩展名），不是靠文件名相等
    2. 读序列   NumberOfFrames 这个 tag 是 IS（ASCII）类型，得用 pydicom 解析；
                自己按 uint16 去读会得到 8248（"8" + 两个填充字节）
    3. MinIP    血管在 DSA 里**偏暗**，所以取 min 投影才能得到完整血管树
                （实测 max 投影会把血管糊掉）
    4. 轴序纠偏 🔴 2026-09-26 更正：label nii **一律**按 (cols, rows) 存，
                非正方形样本（472×512 / 960×742，共 28 例）形状就能看出来是转置的，
                但**正方形样本 label.shape 恰好等于 (rows, cols)**，"形状不符才转置"
                这条判据会静默放行 → 196/224 例（87.5%）标签被整体错位 90° 喂进网络。
                代价：5 折 test 两类平均 Dice 被锁死在 0.1675，而判据生效的 27 例
                样本 Dice 均值 0.73（27/27 vs 0/161，完美分离）。
                ✅ 现在改成「对 8 种二面体变换各算一次标签-图像对齐 AUC，取最大者」，
                不依赖形状。详见 fix_label_axis()。
    5. 尺寸统一 原始有 7 种分辨率（472×512 ~ 1432×1432）。
                做法：**长边缩放到 512，再中心 pad 到 512×512** —— 保持长宽比、
                不丢内容、不引入形变。直接 resize 到 512 会让解剖比例失真。
    6. 落盘     seq 与 minip 存 uint16（保留原始 12-bit 精度，训练时再归一化），
                label 存 uint8

用法：
    E:\\Anaconda\\envs\\dsca\\python.exe data\\prepare.py
    E:\\Anaconda\\envs\\dsca\\python.exe data\\prepare.py --size 512 --folds 5 --seed 0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from skimage.transform import resize as sk_resize

PREFIX_RE = re.compile(r"^(pro|fu)_")
SUFFIXES = (".nii.gz", ".nii", ".dcm")


# ----------------------------------------------------------------------------
def core_of(p: Path) -> str:
    """pro_CXL_AP_38.dcm -> CXL_AP_38 ；fu_CXL_AP_38.nii.gz -> CXL_AP_38"""
    name = p.name
    for s in SUFFIXES:
        if name.endswith(s):
            name = name[: -len(s)]
            break
    return PREFIX_RE.sub("", name)


def patient_of(core: str) -> str:
    """CXL_AP_38 -> CXL。用于按病人分组划分折，避免同一病人跨折造成泄漏。"""
    return core.split("_")[0] if core else core


# ----------------------------------------------------------------------------
def load_dicom_seq(path: Path) -> tuple[np.ndarray, dict]:
    """返回 (T, R, C) uint16 与若干元信息。"""
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None, ...]
    meta = {
        "frames": int(arr.shape[0]),
        "rows": int(arr.shape[1]),
        "cols": int(arr.shape[2]),
        "bits": int(getattr(ds, "BitsStored", 0) or 0),
        "slope": float(getattr(ds, "RescaleSlope", 1) or 1),
        "intercept": float(getattr(ds, "RescaleIntercept", 0) or 0),
    }
    return arr.astype(np.uint16), meta


def load_label(path: Path) -> np.ndarray:
    return np.squeeze(np.asarray(nib.load(str(path)).dataobj)).astype(np.uint16)


def _align_auc(lab: np.ndarray, img: np.ndarray, margin: int = 40,
               max_side: int = 256) -> float:
    """标签-图像对齐度：AUC = P(-img 在前景 > -img 在背景)。

    血管在 DSA 里**偏暗**，所以用 -img 当分数。ROI 取 label 外接框 + margin，
    再整幅降采样到 max_side 以内（对齐度是粗尺度性质，降采样不影响判读但快 ~10 倍）。
    1.0 = 完美对齐；0.5 = 完全无信息（说明标签和图像不是同一张）。
    """
    ys, xs = np.where(lab > 0)
    if len(ys) < 50:
        return float("nan")
    y0, y1 = max(0, ys.min() - margin), min(lab.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(lab.shape[1], xs.max() + margin + 1)
    L = (lab[y0:y1, x0:x1] > 0)
    S = -img[y0:y1, x0:x1].astype(np.float64)
    step = max(1, int(np.ceil(max(L.shape) / max_side)))
    if step > 1:
        L, S = L[::step, ::step], S[::step, ::step]
    f, b = S[L], S[~L]
    if len(f) < 20 or len(b) < 20:
        return float("nan")
    allv = np.concatenate([f, b])
    if allv.size < 2 or allv.std() == 0:
        return float("nan")
    r = allv.argsort().argsort()                    # 秩，避免量纲影响
    return float((r[: len(f)].mean() - r[len(f):].mean()) / allv.size + 0.5)


def _dihedral_variants(label: np.ndarray, want: tuple[int, int]) -> dict[str, np.ndarray]:
    """8 种二面体变换，只保留变换后形状 == want（即 (rows, cols)）的那些。

    非正方形输入时只有 4 个转置类变体形状合法；正方形时 8 个全部合法 —— 这正是
    原来那条形状判据失效的根源，所以这里必须显式枚举再靠证据挑。
    """
    cands = {
        "I":           label,
        "T":           label.T,
        "flipud":      label[::-1, :],
        "fliplr":      label[:, ::-1],
        "T+flipud":    label.T[::-1, :],
        "T+fliplr":    label.T[:, ::-1],
        "rot180":      label[::-1, ::-1],
        "T+rot180":    label.T[::-1, ::-1],
    }
    return {k: np.ascontiguousarray(v) for k, v in cands.items() if v.shape == want}


def fix_label_axis(label: np.ndarray, rows: int, cols: int,
                   ref_img: np.ndarray | None = None, mode: str = "align",
                   margin: int = 40) -> tuple[np.ndarray, str, dict]:
    """把 label 的方向对齐到 DICOM 的 (Rows, Cols)。返回 (label, 方向标记, 判定信息)。

    🔴 不要用形状判断方向（2026-09-26 实测踩坑）
        DSCA 的 label nii **一律**按 (cols, rows) 存：非正方形样本看形状就知道
        （742×960 vs DICOM 960×742），但正方形样本 label.shape 正好等于
        (rows, cols)，于是"形状不符才转置"的判据静默放行 —— 224 例里 196 例
        （87.5%）带的是**转置过的**标签，与图像整体错位 90°。
        后果：5 折 test 两类平均 Dice 恒在 0.1675 附近，其中判据生效的 27 例
        Dice 均值 0.73、未生效的 161 例均值 0.06（零例外）—— 完美分离，
        说明瓶颈是数据管线而不是模型/泛化。
        更坑的是**单样本过拟合查不出来**：转置是确定性几何映射，网络对单张图
        可以硬背，实测错位的 CXL_AP_38 单样本 300 iter 也能到 BV0.62/MAT0.97。

    mode="align" （默认）枚举二面体变换、取标签-图像对齐 AUC 最大者；
    mode="shape" 复现旧的形状判据（仅供 A/B 对照，别用于正式训练）。
    """
    want = (rows, cols)
    shape_rule = "I" if label.shape == want else ("T" if label.shape == (cols, rows) else "resize")
    info = {"shape_rule": shape_rule, "mode": mode, "auc": float("nan"),
            "margin": float("nan"), "variant": shape_rule, "aucs": {}}

    if mode == "shape":
        if label.shape == want:
            return label, "I", info
        if label.shape == (cols, rows):
            return np.ascontiguousarray(label.T), "T", info
        fixed = sk_resize(label.astype(np.float32), want, order=0,
                          preserve_range=True, anti_aliasing=False)
        return fixed.astype(np.uint16), f"resize", info

    cands = _dihedral_variants(label, want)
    if ref_img is not None and cands:
        aucs = {k: _align_auc(v, ref_img, margin=margin) for k, v in cands.items()}
        info["aucs"] = {k: round(v, 4) for k, v in aucs.items() if v == v}
        ok = {k: v for k, v in aucs.items() if v == v}
        if ok:
            best = max(ok, key=ok.get)
            second = max((v for k, v in ok.items() if k != best), default=float("nan"))
            info.update(auc=ok[best], variant=best,
                        margin=float(ok[best] - second) if second == second else float("nan"))
            return cands[best], best, info

    # 没有参考图 / 全部不可判 -> 退回形状判据
    if label.shape == want:
        return label, "I", info
    if label.shape == (cols, rows):
        return np.ascontiguousarray(label.T), "T", info
    fixed = sk_resize(label.astype(np.float32), want, order=0,
                      preserve_range=True, anti_aliasing=False)
    return fixed.astype(np.uint16), "resize", info


# ----------------------------------------------------------------------------
def fit_to_square(vol: np.ndarray, size: int, order: int) -> np.ndarray:
    """(H,W) 或 (T,H,W) -> 长边缩放到 size，再中心 pad 成 (size,size)。

    order=1 用于图像（双线性），order=0 用于标签（最近邻，避免产生中间类别）。
    """
    single = vol.ndim == 2
    if single:
        vol = vol[None, ...]
    t, h, w = vol.shape
    scale = size / max(h, w)
    nh = min(size, max(1, int(round(h * scale))))
    nw = min(size, max(1, int(round(w * scale))))

    out = np.zeros((t, size, size), dtype=np.float32)
    top, left = (size - nh) // 2, (size - nw) // 2
    for i in range(t):
        r = sk_resize(vol[i].astype(np.float32), (nh, nw), order=order,
                      preserve_range=True, anti_aliasing=(order > 0))
        out[i, top:top + nh, left:left + nw] = r
    return out[0] if single else out


# ----------------------------------------------------------------------------
def make_folds(cores: list[str], n_folds: int, seed: int) -> dict:
    """按**病人**分组做 5 折，避免同一病人的不同投照角度跨折泄漏。

    分配用贪心：把"大病人组"先放进当前样本数最少的折。
    直接用 i % n_folds 轮流丢会导致各折极不均衡（实测最大组会把某一折撑爆，
    出现 val 23 / val 50 这种 2 倍差距，让 5 折结果没法互相比较）。
    """
    rng = np.random.RandomState(seed)
    groups: dict[str, list[str]] = {}
    for c in cores:
        groups.setdefault(patient_of(c), []).append(c)

    items = [(p, sorted(v)) for p, v in groups.items()]
    rng.shuffle(items)
    items.sort(key=lambda kv: -len(kv[1]))            # 组大的先分配

    buckets: list[list[str]] = [[] for _ in range(n_folds)]
    for _, members in items:
        target = min(range(n_folds), key=lambda i: len(buckets[i]))
        buckets[target].extend(members)

    folds = {}
    for k in range(n_folds):
        val = sorted(buckets[k])
        train = sorted(c for j, b in enumerate(buckets) if j != k for c in b)
        folds[f"fold_{k}"] = {"train": train, "val": val}
    return folds


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="DSCA 数据预处理")
    ap.add_argument("--src", default=r"E:\Datasets\DSCA\DSA_public",
                    help="原始数据根目录（含 imagesTr/labelsTr/imagesTs/labelsTs）")
    ap.add_argument("--dst", default=r"E:\Datasets\DSCA\processed",
                    help="输出目录")
    ap.add_argument("--size", type=int, default=512, help="统一后的边长")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 例（调试用）")
    ap.add_argument("--axis-mode", default="align", choices=["align", "shape"],
                    help="label 方向判据：align=按标签-图像对齐 AUC 选（默认，正确）；"
                         "shape=旧的形状判据（有 bug，仅作 A/B 对照）")
    ap.add_argument("--folds_only", action="store_true",
                    help="不处理图像，只根据已生成的 npz 重算 5 折划分")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    # 只重算折划分（改了划分策略后不必重跑几分钟的预处理）
    if args.folds_only:
        train_dir = dst / "train"
        cores = sorted(p.stem for p in train_dir.glob("*.npz"))
        if not cores:
            print(f"[ERROR] {train_dir} 下没有 npz，先跑完整预处理")
            return 2
        folds = make_folds(cores, args.folds, args.seed)
        with open(dst / "folds.json", "w", encoding="utf-8") as f:
            json.dump(folds, f, ensure_ascii=False, indent=2)
        sizes = [len(v["val"]) for v in folds.values()]
        print(f"5 折划分已重算 -> {dst / 'folds.json'}")
        print(f"  样本 {len(cores)} 例，病人 {len({patient_of(c) for c in cores})} 组")
        for k, v in folds.items():
            print(f"  {k}: train {len(v['train'])} / val {len(v['val'])}")
        print(f"  各折 val 规模 {sizes}，极差 {max(sizes)-min(sizes)}")
        return 0

    if not src.is_dir():
        print(f"[ERROR] 原始数据目录不存在: {src}")
        return 2

    report: dict = {"src": str(src), "dst": str(dst), "size": args.size,
                    "splits": {}, "problems": []}
    t0 = time.time()

    for split, img_sub, lab_sub in (("train", "imagesTr", "labelsTr"),
                                    ("test", "imagesTs", "labelsTs")):
        img_dir, lab_dir = src / img_sub, src / lab_sub
        if not img_dir.is_dir() or not lab_dir.is_dir():
            print(f"[WARN] 跳过 {split}：{img_dir} 或 {lab_dir} 不存在")
            continue

        imgs = {core_of(p): p for p in sorted(img_dir.glob("*.dcm"))}
        labs = {core_of(p): p for p in sorted(lab_dir.glob("*.nii.gz"))}
        only_img = sorted(set(imgs) - set(labs))
        only_lab = sorted(set(labs) - set(imgs))
        cores = sorted(set(imgs) & set(labs))
        if args.limit:
            cores = cores[: args.limit]

        out_dir = dst / split
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 74)
        print(f"[{split}] 图像 {len(imgs)} / 标签 {len(labs)} / 可配对 {len(cores)}")
        if only_img:
            print(f"  仅图像无标签 {len(only_img)}: {only_img[:5]}")
        if only_lab:
            print(f"  仅标签无图像 {len(only_lab)}: {only_lab[:5]}")
        print("=" * 74)

        stats = Counter()
        resolutions = Counter()
        class_hist = np.zeros(4, dtype=np.int64)
        transposed = []
        changed = []
        ambiguous = []
        problems = []
        samples = []

        for i, core in enumerate(cores, 1):
            try:
                seq, meta = load_dicom_seq(imgs[core])
                label_raw = load_label(labs[core])
                # 方向判定的参考图 = 该样本自己的 MinIP（也就是模型看到的那张）
                ref = seq.min(axis=0).astype(np.float32)
                label, how, ax = fix_label_axis(label_raw, meta["rows"], meta["cols"],
                                                ref_img=ref, mode=args.axis_mode)
                if how.startswith("T"):
                    transposed.append(core)
                if ax["shape_rule"] != how:
                    changed.append(f"{core}: {ax['shape_rule']} -> {how}")
                if ax["auc"] == ax["auc"] and ax["auc"] < 0.85:
                    ambiguous.append(f"{core}: best={how} auc={ax['auc']:.3f}")
                stats[f"axis_{how}"] += 1

                resolutions[f'{meta["rows"]}x{meta["cols"]}'] += 1

                # 统一尺寸：图像与标签用**同一套几何参数**，保证仍对齐
                seq_f = fit_to_square(seq, args.size, order=1)
                lab_f = fit_to_square(label, args.size, order=0)
                minip_f = seq_f.min(axis=0)                 # 血管偏暗 -> 用 min 投影

                seq_u16 = np.clip(np.rint(seq_f), 0, 65535).astype(np.uint16)
                minip_u16 = np.clip(np.rint(minip_f), 0, 65535).astype(np.uint16)
                lab_u8 = np.clip(np.rint(lab_f), 0, 255).astype(np.uint8)

                if lab_u8.max() > 3:
                    problems.append(f"{core}: 统一后 label 出现 >3 的类别值 {int(lab_u8.max())}")

                np.savez_compressed(out_dir / f"{core}.npz",
                                    seq=seq_u16, minip=minip_u16, label=lab_u8)

                vals, cnts = np.unique(lab_u8, return_counts=True)
                for v, c in zip(vals.tolist(), cnts.tolist()):
                    if v < len(class_hist):
                        class_hist[v] += c
                samples.append({"core": core, "frames": meta["frames"],
                                "orig": f'{meta["rows"]}x{meta["cols"]}',
                                "label_axis": how,
                                "shape_rule_would_be": ax["shape_rule"],
                                "axis_auc": round(ax["auc"], 4) if ax["auc"] == ax["auc"] else None,
                                "axis_margin": round(ax["margin"], 4) if ax["margin"] == ax["margin"] else None,
                                "malign_px": int((lab_u8 > 0).sum()),
                                "bits": meta["bits"]})
            except Exception as e:                                   # noqa: BLE001
                problems.append(f"{core}: {type(e).__name__}: {e}")
                print(f"  [FAIL] {core}: {type(e).__name__}: {e}")

            if i % 20 == 0 or i == len(cores):
                print(f"  处理 {i}/{len(cores)}  ({time.time()-t0:.0f}s)", flush=True)

        aucs = [s["axis_auc"] for s in samples if s["axis_auc"] is not None]
        report["splits"][split] = {
            "n_paired": len(cores),
            "n_saved": len(samples),
            "resolutions": dict(resolutions),
            "axis_mode": args.axis_mode,
            "label_axis": dict(stats),
            "transposed_count": len(transposed),
            "changed_vs_shape_rule": len(changed),
            "axis_auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
            "axis_auc_min": round(float(np.min(aucs)), 4) if aucs else None,
            "axis_auc_lt_0.85": ambiguous,
            "class_pixel_hist": class_hist.tolist(),
            "samples": samples,
        }
        if problems:
            report["problems"].extend([f"[{split}] {p}" for p in problems])

        ratio = class_hist / max(1, class_hist.sum())
        print()
        print(f"  已保存到 {out_dir}")
        print(f"  原始分辨率分布 : {dict(resolutions)}")
        print(f"  方向判据       : {args.axis_mode}（shape=旧 bug 版）")
        print(f"  方向处置分布   : {dict(stats)}")
        print(f"  转置类处置     : {len(transposed)} 例")
        print(f"  🔴 与旧形状判据不同 : {len(changed)} 例  ← 这就是之前被静默放行的样本")
        if aucs:
            print(f"  对齐 AUC       : 均值 {np.mean(aucs):.4f} / 最小 {np.min(aucs):.4f}"
                  f" / <0.85 的 {len(ambiguous)} 例")
        print(f"  像素类别占比   : 背景 {ratio[0]*100:.1f}%  "
              f"BV {ratio[1]*100:.1f}%  MAT {ratio[2]*100:.1f}%")
        if problems:
            print(f"  ⚠️ 有问题 {len(problems)} 例（见 report.json）")

    # 5 折划分（只对 train 划分；test 是官方独立测试集）
    train_cores = [s["core"] for s in report["splits"].get("train", {}).get("samples", [])]
    if train_cores:
        folds = make_folds(train_cores, args.folds, args.seed)
        with open(dst / "folds.json", "w", encoding="utf-8") as f:
            json.dump(folds, f, ensure_ascii=False, indent=2)
        n_pat = len({patient_of(c) for c in train_cores})
        print()
        print("=" * 74)
        print(f"5 折划分已写入 {dst / 'folds.json'}")
        print(f"  样本 {len(train_cores)} 例，病人 {n_pat} 组（按病人分组，防同病人跨折泄漏）")
        for k, v in folds.items():
            print(f"  {k}: train {len(v['train'])} / val {len(v['val'])}")

    with open(dst / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 74)
    print(f"完成，用时 {time.time()-t0:.0f}s")
    print(f"自检报告: {dst / 'report.json'}")
    if report["problems"]:
        print(f"⚠️ 共 {len(report['problems'])} 条问题记录，请查看报告")
        for p in report["problems"][:10]:
            print(f"   - {p}")
    else:
        print("✅ 无问题记录")
    return 0


if __name__ == "__main__":
    sys.exit(main())
