"""学习曲线出图：训练集规模 vs 泛化（36 / 72 / 108 / 144 例）。

回答的问题：训练集规模对泛化有没有「可分辨」的影响？

🔴 本脚本内建判读规则，因为 val 只有 36 例 —— 单次验证本身就是一次抽样。实测：
     · 曲线内相邻点跳变：中位 **0.011**、p95 0.018~0.051
       （f0 的 ep110 −0.043 / ep220 −0.082 是尾部单点事件，不是典型值）
     · **同条件跨折离散**：同样是 144 例、只换折划分，末段 val 的
       std = **0.078**、极差 **0.186**
   ⇒ 比较「不同数据量 / 不同配置」时，门槛要用**跨折离散（≈0.08）**，
     而不是曲线内抖动（那只是下限）。门槛之内一律不许解释成「有差别」。
   门槛来源：--band 直接给；--band-from 用同条件重复实验现算；否则退化为曲线内抖动。

用法（本机即可，纯 CPU，不用 GPU）：
    python plot_learning_curve.py                          # 零参数，自动找下面两种摆放
    python plot_learning_curve.py --smooth 10 --tail 100
    python plot_learning_curve.py --spec "a=histories/lc_n36.json" "ref=histories/f0.json"

数据怎么摆（两种都支持，按顺序找，找到就用）：
    A) 平铺   histories/lc_n36.json  lc_n72.json  lc_n108.json  f0.json
              ← 从 SFTP/MobaXterm 拖单个文件最省事，推荐
    B) 归位   runs/lc_n36/history.json  runs/5fold_adam_f0/history.json  ...
              ← 带 train.log / best.pth 的完整目录

    ⚠️ 只需要 history.json（几十 KB），.pth 一律不用传。

产物：
    learning_curve.png    四条曲线 + 噪声带阴影 + 脱困点
    learning_curve.csv    逐点数据（原始 + 平滑），直接贴进汇报

⚠️ 图内文字一律用英文：集群节点常缺中文字体，中文会渲染成方块。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent

# 默认 spec：图例名 = 训练集例数，让图能直接读出结论。
# 值是一个**候选路径列表**，按顺序取第一个存在的 —— 两种摆放都能零参数跑：
#   A) 平铺  histories/<run>.json      ← 从 SFTP 拖文件最省事，推荐
#   B) 归位  runs/<run>/history.json   ← 带 train.log / best.pth 的完整目录
DEFAULT_SPEC = [
    ("n=36   (lc_n36)", ["histories/lc_n36.json", "runs/lc_n36/history.json"]),
    ("n=72   (lc_n72)", ["histories/lc_n72.json", "runs/lc_n72/history.json"]),
    ("n=108  (lc_n108)", ["histories/lc_n108.json", "runs/lc_n108/history.json"]),
    ("n=144  (ref f0)", ["histories/f0.json", "runs/5fold_adam_f0/history.json"]),
]
REF_NAME = "n=144  (ref f0)"     # 噪声带参考（它跑满了 500 epoch）
ESCAPE_TH = 0.05                 # 「逃出全背景平凡解」阈值
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def load_run(path: Path):
    """读 history.json -> (epoch[], val_mean_foreground[])，只取有验证的 epoch。"""
    f = path if path.is_file() else path / "history.json"
    if not f.exists():
        return None
    hist = json.loads(f.read_text(encoding="utf-8"))
    ep, vf = [], []
    for r in hist:
        vd = r.get("val_dice")
        if isinstance(vd, dict) and "mean_foreground" in vd:
            ep.append(int(r["epoch"]))
            vf.append(float(vd["mean_foreground"]))
    return np.asarray(ep, float), np.asarray(vf, float)


def smooth(y, w):
    """居中滑动平均；端点 edge padding，输出长度与输入一致。"""
    if w <= 1 or y.size < 2:
        return y.copy()
    w = min(int(w), y.size)
    pad = w // 2
    yp = np.pad(y, pad, mode="edge")
    return np.convolve(yp, np.ones(w) / w, mode="valid")[: y.size]


def estimate_band(series_list, w):
    """曲线内抖动（稳健估计）—— 用**二阶差分**，不用「去趋势残差」。

        sigma = median(|y[i+1] - 2*y[i] + y[i-1]|) / (1.4826 * sqrt(6))

    为什么换掉旧做法：旧做法用「y 减去滑动均值」的残差，而那个均值里**含 y 本身**，
    窗口越小偏置越大 —— 实测 n=50 / w=10 时把 sigma 低估约 2 倍
    （报 0.0097，而相邻点 p95 跳变其实是 0.051）。二阶差分对线性趋势不敏感、
    且完全不涉及窗口，没有这个偏置。

    🔴 但它只是**曲线内抖动**，是判读门槛的**下限**，不是门槛本身。
       要判断「不同数据量 / 不同配置」的差别，正确门槛是**同条件下的跨实验离散**：
       本项目实测 —— 同样是 144 例、只换折划分，末段 val 的 std = **0.078**、极差 **0.186**。
       有这类重复实验时请用 --band-from 指定，或 --band 直接给值。
    """
    sig, jumps = [], []
    for y in series_list:
        if y.size >= 5:
            s = np.diff(np.diff(y))
            sig.append(1.4826 * float(np.median(np.abs(s - np.median(s)))) / np.sqrt(6.0))
        if y.size >= 2:
            jumps.append(float(np.percentile(np.abs(np.diff(y)), 95)))
    sigma = float(np.median(sig)) if sig else float("nan")
    jump95 = float(np.max(jumps)) if jumps else float("nan")
    return sigma, jump95


def band_from_tails(series, names, tail):
    """跨实验离散 = 各条曲线「末段均值」的样本标准差 —— 这才是判读门槛。

    只在传入的是**同条件重复实验**（同数据量、同超参，只换折/换种子）时才有意义。
    """
    vals = []
    for n in names:
        vf = series[n][1]
        vals.append((n, float(vf[-max(1, min(tail, vf.size)):].mean())))
    if len(vals) < 2:
        return float("nan"), vals
    return float(np.std([v for _, v in vals], ddof=1)), vals


def escape_epoch(ep, y, th=ESCAPE_TH):
    idx = np.where(y > th)[0]
    return int(ep[idx[0]]) if idx.size else None


def main():
    ap = argparse.ArgumentParser(description="learning curve plot")
    ap.add_argument("--runs-root", default=str(_HERE))
    ap.add_argument("--spec", nargs="*", default=None, help="name=path ...")
    ap.add_argument("--smooth", type=int, default=10, help="滑动平均窗口（验证点个数）")
    ap.add_argument("--ref", default=REF_NAME,
                    help="哪条曲线用来估曲线内抖动（默认 144 例参考线）")
    ap.add_argument("--band", type=float, default=0.0,
                    help="直接指定判读门槛 sigma（>0 时优先于其它估计）")
    ap.add_argument("--band-from", default="",
                    help="用这些曲线「末段均值」的标准差当门槛，逗号分隔。"
                         "只应传**同条件的重复实验**（同数据量、同超参，只换折/换种子）")
    ap.add_argument("--tail", type=int, default=100, help="末段均值取多少 epoch")
    ap.add_argument("--out", default="learning_curve.png")
    ap.add_argument("--title", default=None,
                    help="覆盖图标题（做本机渲染自检时用，避免标题与实际内容不符）")
    args = ap.parse_args()

    root = Path(args.runs_root)
    if args.spec:
        spec = []
        for s in args.spec:
            if "=" not in s:
                print("[ERROR] --spec 要 name=path 形式:", s)
                return 2
            k, v = s.split("=", 1)
            spec.append((k.strip(), [v.strip()]))
    else:
        spec = DEFAULT_SPEC

    series = {}
    print("=" * 96)
    print("  %-18s %5s %8s %5s %6s %12s %7s   %s"
          % ("run", "#val", "best", "@ep", "lastep", "tail%d mean" % args.tail,
             "escape", "source"))
    print("-" * 96)
    for name, cands in spec:
        p = None
        for c in cands:
            q = Path(c)
            if not q.is_absolute():
                q = root / c
            if q.exists():
                p = q
                break
        if p is None:
            print("  %-18s   [skip] 找不到：%s" % (name, "  或  ".join(cands)))
            continue
        out = load_run(p)
        if out is None:
            print("  %-18s   [skip] 这个路径存在但不是文件也不是目录：%s" % (name, p))
            continue
        ep, vf = out
        if ep.size == 0:
            print("  %-18s   [skip] 还没有任何验证点（未到 --val-every）" % name)
            continue
        series[name] = (ep, vf)
        k = max(1, min(args.tail, vf.size))
        print("  %-18s %5d %8.4f %5d %6d %12.4f %7s   %s"
              % (name, ep.size, vf.max(), int(ep[vf.argmax()]), int(ep[-1]),
                 float(vf[-k:].mean()), escape_epoch(ep, vf) or "-", p))

    if not series:
        print("[ERROR] 一条可用曲线都没有：检查 --runs-root / --spec")
        return 1
    if len(series) > 1 and min(v.size for _, v in series.values()) < 20:
        print("\n[WARN] 有的曲线验证点不足 20 个（实验还在跑）→ 此刻任何排序都不可信。")

    if args.ref in series:
        band_src = [series[args.ref][1]]
        ref_note = "参考曲线 %s" % args.ref
    else:
        band_src = [v for _, v in series.values()]
        ref_note = "全部曲线（没找到参考 %r）" % args.ref
    sigma_self, jump95 = estimate_band(band_src, args.smooth)

    print("-" * 96)
    print("  曲线内抖动 sigma_self = %s    相邻点跳变 p95 = %s    [%s]"
          % ("n/a" if not np.isfinite(sigma_self) else "%.4f" % sigma_self,
             "n/a" if not np.isfinite(jump95) else "%.4f" % jump95, ref_note))

    if args.band > 0:
        sigma = args.band
        band_note = "手工指定 --band"
    elif args.band_from.strip():
        bf = [s.strip() for s in args.band_from.split(",") if s.strip() in series]
        sigma, vals = band_from_tails(series, bf, args.tail)
        band_note = "跨实验离散（%s 的末段均值 std，n=%d）" % (", ".join(bf), len(bf))
        for n, v in vals:
            print("      %-18s 末段均值 %.4f" % (n, v))
    else:
        sigma = sigma_self
        band_note = "曲线内抖动 —— 🔴 这是**下限**，不是判读门槛"

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.05
        print("\n[WARN] 门槛估计失败，退化为经验值 0.0500")

    print("  ⇒ 判读门槛 sigma = %.4f    [%s]" % (sigma, band_note))
    if band_note.startswith("曲线内"):
        print("     ⚠️ 比较「不同数据量 / 不同配置」时这个门槛偏小：本项目实测，")
        print("        同样是 144 例、只换折划分，末段 val 的 std 就是 0.078（极差 0.186）。")
        print("        有重复实验时请用 --band-from 指定，那才是真正的门槛。")
    print("     判读规则：曲线差距 < %.4f 一律按「不可分辨」处理" % sigma)

    # ---------------- 两两比较：先检查进度是否齐 ----------
    # 🔴 各条曲线的最后 epoch 差得多时，「末段均值」是拿 135 轮比 500 轮，结论必错。
    #    此时改用「同 epoch 对齐」：取全部曲线都有验证点的最大 epoch 做比较。
    names = list(series)
    last_eps = {n: float(series[n][0][-1]) for n in names}
    max_last = max(last_eps.values())
    misaligned = (len(names) > 1) and (max_last - min(last_eps.values()) > 0.10 * max_last)

    common = None
    if len(names) > 1:
        inter = set(series[names[0]][0].tolist())
        for n in names[1:]:
            inter &= set(series[n][0].tolist())
        if inter:
            common = int(max(inter))

    metric = None
    if misaligned:
        print("\n  ⚠️ 进度不一致（最后 epoch：%s）→ 末段均值口径不可比。"
              % "  ".join("%s=%d" % (n, last_eps[n]) for n in names))
        if common is None:
            print("     且没有共同的验证 epoch → 本次不给两两判定，只列末值。")
            for n in names:
                print("    %-18s last_ep=%4d  末值 val %.4f"
                      % (n, last_eps[n], float(series[n][1][-1])))
        else:
            print("     → 改用「同 epoch 对齐」口径：epoch %d（全部曲线都有验证点）" % common)
            metric = {}
            for n in names:
                ep, vf = series[n]
                metric[n] = float(vf[int(np.where(ep == common)[0][0])])
                print("    %-18s ep%d val = %.4f" % (n, common, metric[n]))
    else:
        print("\n  两两差距（末段均值口径，tail=%d）：" % args.tail)
        metric = {}
        for n in names:
            vf = series[n][1]
            metric[n] = float(vf[-max(1, min(args.tail, vf.size)):].mean())
            print("    %-18s last_ep=%4d  末段均值 %.4f" % (n, last_eps[n], metric[n]))

    if metric:
        print("")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                d = abs(metric[a] - metric[b])
                print("    %-18s vs %-18s  Δ=%.4f  ->  %s"
                      % (a, b, d, "不可分辨" if d < sigma else "★ 有差别"))

    # ---------------- 出图 ----------------
    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=120)
    for i, (name, (ep, vf)) in enumerate(series.items()):
        c = COLORS[i % len(COLORS)]
        ax.plot(ep, vf, color=c, lw=0.8, alpha=0.28, marker="o", ms=2.4, zorder=2)
        ax.plot(ep, smooth(vf, args.smooth), color=c, lw=2.3, zorder=4, label=name)
        e = escape_epoch(ep, vf)
        if e is not None:
            k = int(np.where(ep == e)[0][0])
            ax.plot([e], [vf[k]], marker="*", ms=13, color=c, mec="k", mew=0.6, zorder=5)

    if args.ref in series:
        ep_r, vf_r = series[args.ref]
        sm_r = smooth(vf_r, args.smooth)
        ax.fill_between(ep_r, sm_r - sigma, sm_r + sigma, color="0.35", alpha=0.15,
                        zorder=1, label="noise band  +/-%.4f (ref)" % sigma)

    ax.set_xlabel("epoch")
    ax.set_ylabel("val Dice  (mean foreground, val n=36)")
    ax.set_title(args.title or
                 "Learning curve: train-set size vs val Dice   (fold 0, same val set)")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.text(0.012, 0.975,
            "thin line = raw val (single 36-sample draw)\n"
            "thick line = %d-point moving average\n"
            "star = first epoch > %.2f  (escaped all-background solution)\n"
            "rule: |delta| < %.4f  ->  NOT distinguishable"
            % (args.smooth, ESCAPE_TH, sigma),
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.7", alpha=0.92))
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print("\n[ok] 图   -> %s" % Path(args.out).resolve())

    # ---------------- 落 CSV ----------------
    csv_path = Path(args.out).with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "epoch", "val_mean_foreground", "smooth%d" % args.smooth])
        for name, (ep, vf) in series.items():
            sm = smooth(vf, args.smooth)
            for a, b, c in zip(ep.tolist(), vf.tolist(), sm.tolist()):
                w.writerow([name, int(a), "%.6f" % b, "%.6f" % c])
    print("[ok] 数据 -> %s" % csv_path.resolve())

    print("\n汇报时必须同时声明的两条：")
    print("  1) val 仅 36 例，差距 < %.4f 的排序不成立；" % sigma)
    print("  2) val 系统性高于 test（f0: val 0.2590 / test 0.1587，差 0.10），")
    print("     本图结论只能表述为「在 val 口径下」，不能外推 test 绝对值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
