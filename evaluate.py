"""DSCA 评估脚本：算 Dice，逐例输出。

用法（在 F:\\Projects\\DSCA 下）：
    # 在训练时的验证折上评估（默认带镜像 TTA）
    E:\\Anaconda\\envs\\dsca\\python.exe evaluate.py --ckpt runs/fold0_100ep/best.pth --split val --fold 0

    # 在官方独立测试集上评估（44 例）
    E:\\Anaconda\\envs\\dsca\\python.exe evaluate.py --ckpt runs/fold0_100ep/best.pth --split test

    # 关掉 TTA 对比一下 TTA 带来多少提升
    E:\\Anaconda\\envs\\dsca\\python.exe evaluate.py --ckpt ... --split test --no-tta
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.dataset import DEFAULT_NORM, NORM_MODES, DSCADataset        # noqa: E402
from inference import predict_label                       # noqa: E402
from metrics import (CLASS_NAMES, aggregate_dice, aggregate_stats,   # noqa: E402
                     dice_per_class, format_dice, format_stats,
                     stats_per_class)
from models.dsanet import DSANet, MinIPOnly                          # noqa: E402


def build_from_ckpt(ckpt: dict, device):
    targs = ckpt.get("args", {}) or {}
    arch = targs.get("arch", "dsanet")          # 旧 ckpt 无此字段 -> 回落到 dsanet
    if arch == "minip_only":
        model = MinIPOnly(
            grad_checkpoint=False,                 # 推理不需要
            deep_supervision=not targs.get("no_deep_supervision", False),
        )
    else:
        model = DSANet(
            num_frames=targs.get("num_frames", 8),
            tf_layers=targs.get("tf_layers", 4),
            tf_heads=targs.get("tf_heads", 8),
            alpha_mode=targs.get("alpha_mode", "attention"),
            attn_backend=targs.get("attn_backend", "sdpa"),
            grad_checkpoint=False,                 # 推理不需要
            deep_supervision=not targs.get("no_deep_supervision", False),
        )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, targs


def resolve_cores(data_root: Path, split: str, fold: int) -> tuple[str, list[str]]:
    if split == "test":
        cores = sorted(p.stem for p in (data_root / "test").glob("*.npz"))
        return "test", cores
    folds = json.loads((data_root / "folds.json").read_text(encoding="utf-8"))
    key = f"fold_{fold}"
    return "train", folds[key]["val"]


def main() -> int:
    ap = argparse.ArgumentParser(description="DSCA 评估")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=r"E:\Datasets\DSCA\processed")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default=None, help="结果 json 路径（默认放在 ckpt 旁边）")
    ap.add_argument("--norm", default=None, choices=list(NORM_MODES),
                    help="覆盖归一化方式（默认沿用 ckpt 里训练时用的那个）")
    ap.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[ERROR] 找不到权重: {ckpt_path}")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, targs = build_from_ckpt(ckpt, device)

    data_root = Path(args.data)
    sub, cores = resolve_cores(data_root, args.split, args.fold)
    if not cores:
        print(f"[ERROR] {args.split} 下没有样本")
        return 2

    # 🔴 归一化必须与训练时**完全一致**，否则输入分布对不上，Dice 会莫名其妙地低。
    #    优先级：命令行 --norm > ckpt 里存的 args.norm > 默认值（并警告）
    norm = args.norm or targs.get("norm")
    if norm is None:
        norm = DEFAULT_NORM
        print(f"[WARN] ckpt 里没有记录归一化方式，按默认 {norm!r} 评估。"
              f"若权重是用旧的 /4095 训练的，请加 --norm fixed 重跑。")
    ds = DSCADataset(data_root, sub, cores=cores, num_frames=targs.get("num_frames", 8),
                     augment=False, norm=norm)

    print("=" * 78)
    print(f"  评估  ckpt={ckpt_path.name}  epoch={ckpt.get('epoch')}")
    print(f"  集合={args.split}  样本={len(cores)}  TTA={'开' if args.tta else '关'}")
    print("=" * 78)

    all_dice, all_present, per_sample, all_stats = [], [], [], []
    t0 = time.time()
    for i, core in enumerate(cores, 1):
        seq, minip, label = ds[i - 1]
        seq = seq.unsqueeze(0).to(device)
        minip = minip.unsqueeze(0).to(device)
        pred = predict_label(model, seq, minip, use_amp=True, tta=args.tta)[0].cpu().numpy()
        d, p = dice_per_class(pred, label.numpy())
        all_dice.append(d)
        all_present.append(p)
        # 🔴 新增 JAC/SEN/PRE（论文 Table III 报了这五列）。
        #    ⚠️ 下面 BV/MAT/background 三个键**必须保持原样**——eval_calibrate.py
        #    与 tmp/audit_res_vs_dice.py 都按这三个键读，改键名会一起坏。
        st = stats_per_class(pred, label.numpy())
        all_stats.append(st)
        row = {"core": core,
               **{CLASS_NAMES[c]: (None if p[c] is False or d[c] != d[c] else float(d[c]))
                  for c in range(len(CLASS_NAMES))}}
        for name in CLASS_NAMES:
            for k in ("SEN", "PRE", "JAC"):
                v = st[name][k.lower()]
                row[f"{name}_{k}"] = None if v is None or v != v else float(v)
        per_sample.append(row)
        if i % 10 == 0 or i == len(cores):
            print(f"  {i}/{len(cores)}  ({time.time()-t0:.0f}s)", flush=True)

    agg = aggregate_dice(all_dice, all_present)
    stats = aggregate_stats(all_stats)
    print()
    print(f"  前景均值 Dice : {agg['mean_foreground']:.4f}")
    print(f"  三类均值 Dice : {agg['mean_all_classes']:.4f}")
    print(f"  BV  /  MAT    : {agg['per_class']['BV']:.4f} / {agg['per_class']['MAT']:.4f}")
    print(f"  背景          : {agg['per_class']['background']:.4f}")
    print(f"  有效样本      : {agg['n_samples']}")
    print()
    print("  ── JAC / SEN / PRE（论文 Table III 报的其余四列）──")
    print(format_stats(stats))

    out_path = Path(args.out) if args.out else ckpt_path.parent / f"eval_{args.split}.json"
    out_path.write_text(json.dumps(
        {"ckpt": str(ckpt_path), "epoch": ckpt.get("epoch"), "split": args.split,
         "tta": args.tta, "fold": args.fold, "overall": agg,
         "stats": stats, "per_sample": per_sample},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
