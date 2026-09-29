"""标签-图像方向对齐审计（P0 数据管线的验收工具）。

为什么需要这个脚本
------------------
2026-09-26 发现：`data/prepare.py` 原来用「形状不符才转置」判断 label 方向，
而 DSCA 的 label nii **一律**按 (cols, rows) 存。非正方形样本（472×512 / 960×742）
形状就能看出不符 → 被判据救回来了；正方形样本 label.shape 恰好等于 (rows, cols)
→ 判据静默放行，**标签整体错位 90°** 直接进了训练集。224 例里 196 例中招。
后果：5 折 test 两类平均 Dice 被锁死在 0.1675，而判据生效的 27 例均值 0.73
（27/27 vs 0/161，零例外）。

这个脚本用「标签 vs MinIP 暗像素的对齐 AUC」把方向对不对量化出来：
    AUC > 0.9  → 标签与图像一致，可用
    AUC ≈ 0.5  → 标签和图像没关系（错位 / 被转置 / 标错样本）
判据不依赖形状，所以**正方形样本也骗不过它** —— 这正是旧判据的盲区。

用法
----
    # 验收修好的数据
    E:\\Anaconda\\envs\\dsca\\python.exe tmp\\audit_label_align.py --proc "E:\\Datasets\\DSCA\\processed_fix"

    # A/B 对照（新 vs 旧），会列出被纠正的样本
    E:\\Anaconda\\envs\\dsca\\python.exe tmp\\audit_label_align.py ^
        --proc "E:\\Datasets\\DSCA\\processed_fix" --compare "E:\\Datasets\\DSCA\\processed"

只读：只读 npz 里的 minip 与 label，不解压 seq，不改任何数据。
报告写到 <proc>/audit_align.txt。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.prepare import _align_auc  # noqa: E402  与 prepare.py 共用同一套判据


def list_npz(proc: Path) -> list[tuple[str, str, Path]]:
    out = []
    for split in ("train", "test"):
        d = proc / split
        if d.is_dir():
            out += [(split, p.stem, p) for p in sorted(d.glob("*.npz"))]
    return out


def raw_sizes(src: Path) -> dict[str, tuple[int, int]]:
    """core -> 原始 DICOM (Rows, Cols)。

    必须回原始数据取尺寸：processed npz 里 **所有** 样本都被统一成 512×512，
    光看 npz 根本分不出「正方形 / 非正方形」两个 cohort —— 这正是旧判据翻车的原因，
    审计脚本自己也要避免踩同一个坑。
    """
    import pydicom
    out: dict[str, tuple[int, int]] = {}
    for sub in ("imagesTr", "imagesTs"):
        d = src / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.dcm")):
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
            out[p.stem[4:]] = (int(ds.Rows), int(ds.Columns))
    return out


def audit(proc: Path, sizes: dict[str, tuple[int, int]] | None = None) -> list[dict]:
    sizes = sizes or {}
    rows = []
    for split, core, p in list_npz(proc):
        with np.load(p) as z:
            lab = z["label"]
            img = z["minip"].astype(np.float32)
        h, w = lab.shape
        a_same = _align_auc(lab, img)
        a_t = _align_auc(np.ascontiguousarray(lab.T), img)
        # 方向安全性用「相对转置的领先幅度」判：低对比样本的绝对 AUC 会偏低，
        # 但只要明显赢过转置，方向就是确定的。
        margin = (a_same - a_t) if (a_same == a_same and a_t == a_t) else float("nan")
        r0, c0 = sizes.get(core, (h, w))
        rows.append({
            "split": split, "core": core, "shape": (h, w),
            "raw": (r0, c0), "square": r0 == c0,
            "auc": a_same, "auc_T": a_t, "margin": margin,
            "verdict": "OK" if a_same > 0.9 else ("BAD" if margin < 0.15 else "WEAK"),
            "T_better": (a_t == a_t) and (a_same == a_same) and a_t > a_same + 0.05,
        })
    return rows


def emit(rows: list[dict], title: str) -> None:
    print("=" * 92)
    print(f"标签-图像方向对齐审计   [{title}]")
    print("=" * 92)
    aucs = np.array([r["auc"] for r in rows if r["auc"] == r["auc"]])
    marg = np.array([r["margin"] for r in rows if r["margin"] == r["margin"]])
    print(f"样本 {len(rows)} 例 | 对齐 AUC 均值 {aucs.mean():.4f} / 最小 {aucs.min():.4f} / "
          f"中位 {np.median(aucs):.4f}")
    print(f"对转置的领先幅度 | 均值 {marg.mean():.4f} / 最小 {marg.min():.4f}"
          f"   （方向安全性主判据：< 0.15 才算可疑）")
    for thr in (0.95, 0.90, 0.80):
        n = int((aucs < thr).sum())
        print(f"  AUC < {thr:.2f} : {n:>4} 例  ({100.0 * n / len(aucs):>5.1f}%)")
    print()
    print(f"  {'分组':<16}{'n':>5}{'AUC均值':>11}{'AUC最小':>10}{'领先最小':>11}"
          f"{'OK(>0.9)':>11}{'转置更优':>10}")
    for key, want_square in (("正方形", True), ("非正方形", False)):
        g = [r for r in rows if r["square"] == want_square]
        if not g:
            continue
        a = np.array([r["auc"] for r in g if r["auc"] == r["auc"]])
        m = np.array([r["margin"] for r in g if r["margin"] == r["margin"]])
        ok = sum(1 for r in g if r["verdict"] == "OK")
        print(f"  {key:<16}{len(g):>5}{a.mean():>11.4f}{a.min():>10.4f}{m.min():>11.4f}"
              f"{ok:>7}/{len(g):<3}{sum(1 for r in g if r['T_better']):>10}")
    suspect = [r for r in rows if r["verdict"] == "BAD"]
    print()
    if suspect:
        print(f"⚠️ 方向存疑 {len(suspect)} 例（所选方向对转置的领先 < 0.15）—— 需人工复核：")
        for r in sorted(suspect, key=lambda x: x["auc"])[:25]:
            tail = "   ← 转置明显更好，方向错了" if r["T_better"] else ""
            print(f"    {r['core']:<16}{r['split']:<7}原始 {str(r['raw']):<12}"
                  f"AUC {r['auc']:.4f}  转置 {r['auc_T']:.4f}  领先 {r['margin']:+.4f}{tail}")
    else:
        print("✅ 方向全部确定：每一例所选方向都明显优于转置（领先 ≥ 0.15）")

    weak = sorted((r for r in rows if r["verdict"] == "WEAK"), key=lambda x: x["auc"])
    if weak:
        print()
        print(f"ℹ️ 低对比但方向确定 {len(weak)} 例（AUC 0.8~0.9，领先幅度 ≥0.15，"
              f"属图像对比度问题而非方向问题）：")
        for r in weak[:15]:
            print(f"    {r['core']:<16}原始 {str(r['raw']):<12}AUC {r['auc']:.4f}  "
                  f"转置 {r['auc_T']:.4f}  领先 {r['margin']:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="标签-图像方向对齐审计")
    ap.add_argument("--proc", required=True, help="processed 目录（含 train/ 与 test/）")
    ap.add_argument("--src", default=r"E:\Datasets\DSCA\DSA_public",
                    help="原始数据根目录（用于取「正方形 / 非正方形」分组，npz 里看不出来）")
    ap.add_argument("--compare", default="", help="可选对照目录（旧版），做 A/B")
    ap.add_argument("--save", default="", help="报告 txt 路径（默认 <proc>/audit_align.txt）")
    ap.add_argument("--rows", type=int, default=0, help="额外逐例打印 AUC 最低的 N 行")
    ap.add_argument("--focus", default="", help="正则：只详细列出匹配的样本（如 JYQ）")
    args = ap.parse_args()

    proc = Path(args.proc)
    if not proc.is_dir():
        print(f"[ERROR] 目录不存在: {proc}")
        return 2

    sizes = raw_sizes(Path(args.src))
    print(f"原始尺寸表：{len(sizes)} 例（来自 {args.src}）\n")

    buf: list[str] = []
    real = sys.stdout

    class Tee:
        def write(self, s: str) -> None:
            real.write(s)
            buf.append(s)

        def flush(self) -> None:
            real.flush()

    sys.stdout = Tee()                     # type: ignore[assignment]
    try:
        rows = audit(proc, sizes)
        emit(rows, str(proc))

        if args.focus:
            import re
            pat = re.compile(args.focus, re.I)
            hit = [r for r in rows if pat.search(r["core"])]
            print()
            print(f"🔍 --focus {args.focus!r}：{len(hit)} 例")
            print(f"  {'core':<16}{'split':<7}{'原始':<12}{'AUC':>9}{'AUC(T)':>10}{'领先':>9}   判定")
            for r in sorted(hit, key=lambda x: x["auc"]):
                print(f"  {r['core']:<16}{r['split']:<7}{str(r['raw']):<12}"
                      f"{r['auc']:>9.4f}{r['auc_T']:>10.4f}{r['margin']:>9.4f}   {r['verdict']}")

        if args.rows:
            print()
            print(f"  {'core':<16}{'split':<7}{'原始':<12}{'AUC':>9}{'AUC(T)':>10}   判定")
            for r in sorted(rows, key=lambda x: x["auc"])[: args.rows]:
                print(f"  {r['core']:<16}{r['split']:<7}{str(r['raw']):<12}"
                      f"{r['auc']:>9.4f}{r['auc_T']:>10.4f}   {r['verdict']}")

        if args.compare:
            old_rows = audit(Path(args.compare), sizes)
            old_map = {r["core"]: r for r in old_rows}
            o = np.array([r["auc"] for r in old_rows])
            n = np.array([r["auc"] for r in rows])
            healed = sorted((r for r in rows if r["core"] in old_map
                             and r["auc"] - old_map[r["core"]]["auc"] > 0.05),
                            key=lambda x: -x["auc"])
            print()
            print("=" * 92)
            print(f"A/B 对照：旧 {args.compare}")
            print(f"          新 {proc}")
            print("=" * 92)
            print(f"  {'':<10}{'AUC均值':>10}{'AUC<0.9':>10}{'可用':>8}{'方向确定':>10}")
            print(f"  {'旧版':<10}{o.mean():>10.4f}{int((o < 0.9).sum()):>10}"
                  f"{len([r for r in old_rows if r['verdict'] == 'OK']):>8}"
                  f"{len([r for r in old_rows if r['verdict'] in ('OK', 'WEAK')]):>10}")
            print(f"  {'新版':<10}{n.mean():>10.4f}{int((n < 0.9).sum()):>10}"
                  f"{len([r for r in rows if r['verdict'] == 'OK']):>8}"
                  f"{len([r for r in rows if r['verdict'] in ('OK', 'WEAK')]):>10}")
            print(f"\n  方向被纠正的样本: {len(healed)} / {len(rows)}")
            for r in healed[:20]:
                print(f"    {r['core']:<16} AUC {old_map[r['core']]['auc']:.4f} → {r['auc']:.4f}")
    finally:
        sys.stdout = real

    out = Path(args.save) if args.save else proc / "audit_align.txt"
    out.write_text("".join(buf), encoding="utf-8")
    print(f"\n报告已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
