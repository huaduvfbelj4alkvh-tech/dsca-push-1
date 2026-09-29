"""失败样本可视化：一眼判断「数据到头了」还是「模型没学到」。

要回答的三个问题：
  1) 第 1 行的 8 帧，肉眼能看出亮度变化（造影剂流动）吗？
     —— 看不出，说明时序通路拿不到信息。
  2) 第 3 行左图：红(BV)/黄(MAT)轮廓里的血管，肉眼可见吗？
     —— 看不见，说明标签与图像不对应，或信号确实太弱。
  3) 第 3 行右图：前景/背景灰度直方图重叠吗？
     —— 完全重叠，说明逐像素不可分。

用法（命令行进到 F:\\Projects\\DSCA 后）：
    E:\\Anaconda\\envs\\dsca\\python.exe viz_failures.py
    E:\\Anaconda\\envs\\dsca\\python.exe viz_failures.py --cores CXY_L_46,HXY_AL_94
    E:\\Anaconda\\envs\\dsca\\python.exe viz_failures.py --split train --cores CXL_AP_38
    # 带上当次评估的 json，标题才会显示真实的 model Dice：
    E:\\Anaconda\\envs\\dsca\\python.exe viz_failures.py --dice-json runs/f0_axfix/eval_test.json

🔴 2026-09-27 两处修正（都会影响这张图能不能当证据用）：
    1) model Dice 不再硬编码。原来写死了一份 44 例的前景 Dice，实为某次**坏 run**
       （均值≈0.16）的手抄值，会让图上数字与当前权重完全不符。现在从 eval_*.json 读，
       没有就显示 n/a。
    2) contrast 除全图口径外，另给**局部口径**（血管外 r=10 环带）。全图口径会被大面积
       暗区（准直器/未曝光区，12/44 例约占 27% 面积）拉低，同一张图能算出 +0.13 也能算出
       -0.04。实测 44/44 例局部对比度都是 0.47~0.80 的正值 —— 判读请以 local 为准。

若粘贴后报 SyntaxError，先自查有没有混进不可见字符：
    python -c "s=open('viz_failures.py',encoding='utf-8').read();print('U+200B:',s.count(chr(0x200b)))"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data.dataset import normalize                      # noqa: E402
from diagnose import _contrast, _oracle_threshold       # noqa: E402

# 内置对照：3 个做得好 + 4 个做不出来（含 oracle 高但模型做不出来的 GJW_AP_35；
# 具体 Dice 一律以 --dice-json 读到的当次评估为准，不要凭记忆写数字）
DEFAULT_CORES = ["ZQL_L_16", "RXY_L_44", "ZHL_AP_39",
                 "CXY_AP_19", "HXY_AL_94", "CXY_L_60", "GJW_AP_35"]


def find_eval_json(split: str) -> Path | None:
    """在 runs/*/ 下找最近的 eval_<split>.json。"""
    cands = sorted(_HERE.glob(f"runs/*/eval_{split}.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def load_dice(json_path) -> tuple[dict, str | None]:
    """从 evaluate.py 产出的 eval_*.json 读逐例 Dice。

    🔴 2026-09-27 修正：本文件原先硬编码了一份 44 例前景 Dice 字典，是某次**坏 run**
       （44 例均值 ≈0.16）的手抄值。它会让图上标题的数字与当前权重完全不符 ——
       等于把错误性能印在交付图上。现在一律从 eval json 读；**读不到就显示 n/a，绝不猜**。
    """
    if not json_path:
        return {}, None
    p = Path(json_path)
    if not p.exists():
        print(f"[WARN] eval json 不存在: {p}  -> 标题里的 model Dice 显示 n/a")
        return {}, None
    obj = json.loads(p.read_text(encoding="utf-8"))
    return {r.get("core"): r for r in obj.get("per_sample", [])}, p.name


def _local_contrast(proj: np.ndarray, mask: np.ndarray, r: int = 10) -> float:
    """前景 vs **紧邻背景**的对比度（血管偏暗 -> 应为正）。

    ⚠️ diagnose._contrast 用的是全图背景均值，会被大面积暗区（准直器/未曝光区）拉低，
       于是同一张图能算出 +0.13 也能算出 -0.04，是**误导性的**。这里改用血管外 r 像素
       的环带做参考，实测 44/44 例都是 0.47~0.80 的正值。
    """
    m = mask > 0
    if m.sum() == 0:
        return float("nan")
    ring = ndi.binary_dilation(m, iterations=r) & ~m
    if ring.sum() == 0:
        return float("nan")
    fg, bg = proj[m].mean(), proj[ring].mean()
    return float((bg - fg) / max(1e-6, bg))


def show_gray(ax, img, title, lo=None, hi=None):
    a, b = (np.percentile(img, [1, 99]) if lo is None else (lo, hi))
    if b <= a:
        b = a + 1.0
    ax.imshow(img, cmap="gray", vmin=a, vmax=b)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def draw_one(core, root, out_dir, dice_map=None, json_name=None):
    path = root / (core + ".npz")
    if not path.exists():
        print("[skip] not found:", path)
        return
    d = np.load(path)
    seq = d["seq"].astype(np.float32)
    mini = d["minip"].astype(np.float32)
    lab = d["label"].astype(np.int64)

    lo, hi = np.percentile(seq, [1, 99])       # 8 帧共用同一灰度窗
    rng = seq.max(0) - seq.min(0)              # 逐像素帧间极差

    c_glob = _contrast(mini, lab)              # 全图背景（会被暗区污染，仅留作对照）
    c_loc = _local_contrast(mini, lab)         # 紧邻背景（可信的那个）
    oracle = _oracle_threshold(normalize(mini, "zscore"), lab)[0]
    row = (dice_map or {}).get(core) or {}

    def _fmt(key):
        v = row.get(key)
        return "n/a" if v is None else "%.4f" % v

    dtxt = "BV %s / MAT %s" % (_fmt("BV"), _fmt("MAT"))
    if not row:
        dtxt = "n/a (no eval json)"
    src = json_name or "none"

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 8, height_ratios=[1.0, 1.6, 1.2],
                          hspace=0.28, wspace=0.08)

    for i in range(8):
        ax = fig.add_subplot(gs[0, i])
        if i < seq.shape[0]:
            show_gray(ax, seq[i], "frame %d" % i, lo, hi)
        else:
            ax.axis("off")

    ax = fig.add_subplot(gs[1, 0:2]); show_gray(ax, seq[0], "single frame 0")
    ax = fig.add_subplot(gs[1, 2:4]); show_gray(ax, mini, "MinIP (model input)")
    ax = fig.add_subplot(gs[1, 4:6]); show_gray(ax, seq.max(0), "MaxIP")

    ax = fig.add_subplot(gs[1, 6:8])
    rm = np.percentile(rng, 99) or 1.0
    ax.imshow(np.clip(rng, 0, rm), cmap="inferno")
    ax.set_title("frame range (max-min) p99=%.0f" % rm, fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(gs[2, 0:4])
    show_gray(ax, mini, "MinIP + label contour")
    ax.contour(lab == 1, levels=[0.5], colors="red", linewidths=1.0)
    ax.contour(lab == 2, levels=[0.5], colors="yellow", linewidths=1.0)

    ax = fig.add_subplot(gs[2, 4:8])
    fg = mini[lab > 0]
    bgm = lab == 0
    ring = ndi.binary_dilation(lab > 0, iterations=10) & bgm if fg.size else np.zeros_like(bgm)
    bg_g = mini[bgm]
    bg_r = mini[ring]
    # ⚠️ 全图背景含大面积暗区（准直器/未曝光区），直方图会被它带偏。
    #    判"前景是否可分"要看**环带背景**（紧邻血管的那圈）。
    if bg_g.size:
        ax.hist(bg_g, bins=120, density=True, alpha=0.35,
                label="bg global n=%d" % bg_g.size)
    if bg_r.size:
        ax.hist(bg_r, bins=120, density=True, alpha=0.65,
                label="bg ring(r=10) n=%d" % bg_r.size)
    if fg.size:
        ax.hist(fg, bins=120, density=True, alpha=0.85,
                label="foreground n=%d" % fg.size)
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.set_title("intensity histogram (log) -- compare fg vs ring", fontsize=9)

    fig.suptitle("%s    model Dice %s [%s]    oracle(pixel) %.4f    "
                 "contrast local %+.3f (global %+.3f)    fg %.2f%%"
                 % (core, dtxt, src, oracle, c_loc, c_glob,
                    100.0 * (lab > 0).mean()), fontsize=12)

    out = out_dir / (core + ".png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("[ok] %s   Dice=%s  oracle=%.4f  contrast(local)=%+.3f (global=%+.3f)"
          % (out, dtxt, oracle, c_loc, c_glob))


def main():
    ap = argparse.ArgumentParser(description="failure-case visualization")
    ap.add_argument("--data", default=r"E:\Datasets\DSCA\processed")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--cores", default="", help="comma separated; empty = builtin 7 cases")
    ap.add_argument("--out", default="viz")
    ap.add_argument("--dice-json", default=None,
                    help="evaluate.py 产出的 eval_*.json；不给则自动找 runs/*/eval_<split>.json。"
                         "🔴 标题里的 model Dice 只来自这个文件，绝不硬编码")
    args = ap.parse_args()

    root = Path(args.data) / args.split
    if not root.exists():
        print("[ERROR] not found:", root)
        return 2
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = _HERE / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    jp = args.dice_json or find_eval_json(args.split)
    dice_map, json_name = load_dice(jp)
    if not dice_map:
        print("[WARN] 没有可用的 eval json -> 图上 model Dice 显示 n/a。"
              "本图**不能**当作性能证据使用。")

    cores = [c.strip() for c in args.cores.split(",") if c.strip()] or DEFAULT_CORES
    print("data:", root)
    print("out :", out_dir)
    print("dice:", json_name or "n/a")
    print("cases:", len(cores))
    for core in cores:
        draw_one(core, root, out_dir, dice_map, json_name)

    print("")
    print("看三个问题：")
    print("  1) row1 的 8 帧有肉眼可见的亮度变化吗？(没有 -> 时序通路无信息)")
    print("  2) row3 左：红/黄轮廓里的血管肉眼可见吗？(看不见 -> 数据到头)")
    print("  3) row3 右：fg 与 **ring(r=10)** 两条直方图重叠吗？"
          "(重叠 -> 紧邻背景下逐像素不可分；只看 global 会因为暗区而误判)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
