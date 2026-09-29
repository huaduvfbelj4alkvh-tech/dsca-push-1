"""分割评估指标。

类别：0 = 背景，1 = BV（分支血管），2 = MAT（主干血管）。
论文报告的指标是 Dice。

处理缺失类别的规则（数据实测发现部分样本只有 0/1、没有 MAT）：
    - 真实标签和预测里都没有这个类  -> 不计入平均（nan）
    - 真实标签没有、但预测里有（假阳性）-> 记为 0，这是要惩罚的错误，不能跳过
"""

from __future__ import annotations

import numpy as np

N_CLASSES = 3
CLASS_NAMES = ("background", "BV", "MAT")
CLASS_NAMES_CN = ("背景", "BV 分支", "MAT 主干")


def dice_per_class(pred: np.ndarray, target: np.ndarray,
                   n_classes: int = N_CLASSES, eps: float = 1e-5
                   ) -> tuple[np.ndarray, np.ndarray]:
    """pred/target: 整数标签数组，形状 (H, W)。

    返回 (dice (C,) float、可能含 nan, present (C,) bool)。
    """
    dice = np.full(n_classes, np.nan, dtype=np.float64)
    present = np.zeros(n_classes, dtype=bool)
    for c in range(n_classes):
        p = pred == c
        t = target == c
        p_sum, t_sum = p.sum(), t.sum()
        if p_sum == 0 and t_sum == 0:
            continue                                   # 双方都没有，不计入
        present[c] = True
        dice[c] = (2.0 * (p & t).sum() + eps) / (p_sum + t_sum + eps)
    return dice, present


def aggregate_dice(all_dice: list[np.ndarray],
                   all_present: list[np.ndarray]) -> dict:
    """跨样本聚合：每个类别取"该类别出现过的样本"上的平均。"""
    if not all_dice:
        return {}
    D = np.stack(all_dice)                              # (N, C)
    P = np.stack(all_present)                           # (N, C)
    out = {"n_samples": int(D.shape[0])}
    per_class = {}
    for c in range(D.shape[1]):
        mask = P[:, c]
        if mask.sum() == 0:
            per_class[CLASS_NAMES[c]] = float("nan")
            continue
        per_class[CLASS_NAMES[c]] = float(np.nanmean(D[mask, c]))
    out["per_class"] = per_class

    # 前景平均（BV 与 MAT）——论文主要报这个
    fg = [per_class[CLASS_NAMES[c]] for c in (1, 2)
          if not np.isnan(per_class[CLASS_NAMES[c]])]
    out["mean_foreground"] = float(np.mean(fg)) if fg else float("nan")
    allc = [v for v in per_class.values() if not np.isnan(v)]
    out["mean_all_classes"] = float(np.mean(allc)) if allc else float("nan")
    return out


def confuse_per_class(pred: np.ndarray, target: np.ndarray,
                      n_classes: int = N_CLASSES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐类混淆计数 (tp, fp, fn)，各 (C,) int64。

    ⚠️ 2026-09-26 新增：论文 Table III/IV 报的是 JAC/Dice/**SEN/PRE**/AUC 五列，
        而我们之前只算 Dice —— 于是"BV 差 5 分"到底差在**漏检(SEN)** 还是
        **过检(PRE)** 上，无法回答。SEN/PRE 是分辨这两种失败模式的最小代价手段。
    """
    tp = np.zeros(n_classes, dtype=np.int64)
    fp = np.zeros(n_classes, dtype=np.int64)
    fn = np.zeros(n_classes, dtype=np.int64)
    for c in range(n_classes):
        p = pred == c
        t = target == c
        tp[c] = int(np.count_nonzero(p & t))
        fp[c] = int(np.count_nonzero(p & ~t))
        fn[c] = int(np.count_nonzero(t & ~p))
    return tp, fp, fn


def stats_per_class(pred: np.ndarray, target: np.ndarray,
                    n_classes: int = N_CLASSES, eps: float = 1e-5) -> dict:
    """逐类 dice / sen / pre / jac / tp,fp,fn / present，返回 {类名: {...}}。

    约定（与 `dice_per_class` 保持一致）：
      - 双方都为空 -> present=False，各指标 None（不计入平均）
      - PRE 的分母 TP+FP=0（模型完全没预测该类）-> **记 0.0**，这是标准约定
        （sklearn/torchmetrics 的 zero_division=0），不能记 1.0 也不能跳过，
        否则"整类漏检"会被算成"精确率满分"。
    """
    tp, fp, fn = confuse_per_class(pred, target, n_classes)
    out: dict = {}
    for c in range(n_classes):
        name = CLASS_NAMES[c]
        if tp[c] + fp[c] + fn[c] == 0:
            out[name] = {"dice": None, "sen": None, "pre": None, "jac": None,
                         "tp": 0, "fp": 0, "fn": 0, "present": False}
            continue
        den_s, den_p = int(tp[c] + fn[c]), int(tp[c] + fp[c])
        out[name] = {
            "dice": float((2.0 * tp[c] + eps) / (2.0 * tp[c] + fp[c] + fn[c] + eps)),
            "jac": float((tp[c] + eps) / (tp[c] + fp[c] + fn[c] + eps)),
            "sen": float((tp[c] + eps) / (den_s + eps)) if den_s > 0 else None,
            "pre": float((tp[c] + eps) / (den_p + eps)) if den_p > 0 else 0.0,
            "tp": int(tp[c]), "fp": int(fp[c]), "fn": int(fn[c]), "present": True,
        }
    return out


def aggregate_stats(all_stats: list[dict]) -> dict:
    """跨样本聚合，**两种口径都给**：

    macro_label : 只在「GT 里有该类」的样本上逐样本平均（= 论文口径，见 README §5.6）
    micro       : 把全数据集的 TP/FP/FN 先汇总再算一次（**与缺类约定无关**，
                  是验证 macro 结论的独立视角 —— 血管像素主导，最能反映"细血管漏没漏"）
    """
    out: dict = {"n_samples": len(all_stats), "macro": {}, "micro": {}}
    eps = 1e-5
    for name in CLASS_NAMES:
        vals = {k: [] for k in ("dice", "sen", "pre", "jac")}
        TP = FP = FN = 0
        for s in all_stats:
            st = s.get(name)
            if not st:
                continue
            TP += st["tp"]
            FP += st["fp"]
            FN += st["fn"]
            if st["tp"] + st["fn"] <= 0:        # GT 没有该类 -> label 口径剔除
                continue
            for k in vals:
                v = st[k]
                if v is not None:
                    vals[k].append(v)

        def _m(xs):
            return float(np.mean(xs)) if xs else float("nan")

        out["macro"][name] = {k: _m(v) for k, v in vals.items()}
        out["macro"][name]["n"] = len(vals["dice"])
        tot = TP + FP + FN
        out["micro"][name] = {
            "dice": (2.0 * TP + eps) / (2.0 * TP + FP + FN + eps) if tot else float("nan"),
            "jac": (TP + eps) / (tot + eps) if tot else float("nan"),
            "sen": (TP + eps) / (TP + FN + eps) if (TP + FN) else float("nan"),
            "pre": (TP + eps) / (TP + FP + eps) if (TP + FP) else 0.0,
            "tp": TP, "fp": FP, "fn": FN,
        }

    fg = [out["macro"][n]["dice"] for n in ("BV", "MAT")
          if not np.isnan(out["macro"][n]["dice"])]
    out["macro"]["mean_foreground"] = float(np.mean(fg)) if fg else float("nan")
    return out


def format_dice(agg: dict) -> str:
    if not agg:
        return "（无有效样本）"
    pc = agg["per_class"]
    return (f"背景 {pc['background']:.4f} | BV {pc['BV']:.4f} | MAT {pc['MAT']:.4f}"
            f"  ||  前景均值 {agg['mean_foreground']:.4f}")


def format_stats(stats: dict) -> str:
    """把 aggregate_stats 的结果打成多行，供 evaluate.py 直接 print。"""
    if not stats:
        return "（无有效样本）"
    lines = []
    for tag, key in (("macro(label 口径)", "macro"), ("micro(像素汇总)", "micro")):
        lines.append(f"  [{tag}]")
        for name in ("BV", "MAT"):
            s = stats[key][name]
            lines.append(f"    {name:<4} Dice {s['dice']:.4f}  JAC {s['jac']:.4f}  "
                         f"SEN {s['sen']:.4f}  PRE {s['pre']:.4f}"
                         + (f"   (n={s['n']})" if "n" in s else
                            f"   (TP {s.get('tp', 0)} / FP {s.get('fp', 0)}"
                            f" / FN {s.get('fn', 0)})"))
    return "\n".join(lines)
