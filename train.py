"""DSCA / DSANet 训练脚本。

------------------------------------------------------------------
论文超参（第 IV-A 节）与本机适配
------------------------------------------------------------------
    论文                                   本机（RTX 3060 Laptop，可用 5136 MiB）
    ------------------------------------   ------------------------------------------
    骨干 nnUNet 风格，瓶颈 c=512            同（chs=32..512，4 次下采样 -> 32×32）
    SGD, momentum 0.99, wd 3e-5             同
    初始 lr 0.01 + polynomial decay         同（power 0.9）
    500 epoch × 100 iter                    同（可先用小规模验证）
    batch = 2 @ 512×512                      batch=1 + 梯度累积 2 步（数学等效）
    AMP                                     开（必须）
    梯度检查点                              开（必须：5194 -> 2574 MiB）
    5 折交叉验证                            同（折划分见 processed/folds.json）
    不做后处理                              同

------------------------------------------------------------------
用法
------------------------------------------------------------------
    cd F:\\Projects\\DSCA

    # 1) 冒烟：确认能跑通、loss 会降（约 3~5 分钟）
    E:\\Anaconda\\envs\\dsca\\python.exe train.py --fold 0 --epochs 3 --iters-per-epoch 20 --out runs/smoke

    # 2) 第一阶段：1 折 100 epoch（约 5~10 小时）
    E:\\Anaconda\\envs\\dsca\\python.exe train.py --fold 0 --epochs 100 --out runs/fold0_100ep

    # 3) 完整：1 折 500 epoch
    E:\\Anaconda\\envs\\dsca\\python.exe train.py --fold 0 --epochs 500 --out runs/fold0_full

    # 4) 5 折全部（建议用脚本串行跑，见 README）
    E:\\Anaconda\\envs\\dsca\\python.exe train.py --fold 1 --epochs 500 --out runs/fold1_full

    # 5) 小样本过拟合探针（2026-09-26 加）
    #    🔴 注意：这个探针**查不出轴序/标签错位类 bug** —— 转置是确定性几何映射，
    #    网络对单张图可以硬背（实测错位的 CXL_AP_38 单样本 300 iter 也能到
    #    BV0.62/MAT0.97）。它的用途是查「容量/优化/梯度路径」类问题。
    E:\\Anaconda\\envs\\dsca\\python.exe train.py --fold 0 --limit-train 5 ^
        --epochs 150 --iters-per-epoch 20 --lr-total-epochs 500 --optimizer adam ^
        --lr 1e-3 --val-every 50 --out runs/probe_overfit5

    # 6) 断点续训（2026-09-26 加，之前中断只能从 epoch 0 重来）
    E:\\Anaconda\\envs\\dsca\\python.exe train.py --fold 0 --epochs 500 ^
        --out runs/fold0_full --resume auto
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.dataset import DEFAULT_NORM, NORM_MODES, DSCADataset          # noqa: E402
from inference import predict_label           # noqa: E402
from metrics import aggregate_dice, dice_per_class, format_dice   # noqa: E402
from models.dsanet import DSANet, MinIPOnly              # noqa: E402
from models.losses import DeepSupervisionLoss, inverse_frequency_weights  # noqa: E402


# ----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def poly_lr(epoch: int, total_epochs: int, lr0: float, power: float) -> float:
    """多项式衰减（论文只写 "polynomial decay"，这里取常见的 power=0.9）。

    🔴 `total_epochs` 是**学习率曲线的总长度**，不一定是本次实际跑的 epoch 数 ——
    这一点很容易踩坑，务必看下面的实测对照：

        曲线总长 = 500（论文）       epoch 12 时 lr 还剩 97.8%
        曲线总长 = 12 （短 probe）   epoch 12 时 lr 只剩 10.7%   ← 压缩了 40 倍

    也就是说，直接 `--epochs 12` 做 probe，等于**一上来就把学习率衰减到接近归零**，
    测出来的 Dice 会严重低估模型能力（把"还没学会"误判成"学不会"）。
    → 短周期 probe 必须配 `--lr-total-epochs 500`，让 lr 语义与论文对齐。
    """
    return lr0 * max(0.0, 1.0 - epoch / max(1, total_epochs)) ** power


def build_model(args):
    """按 --arch 选择网络：dsanet = 论文主模型，minip_only = Table VI 基线。"""
    if args.arch == "minip_only":
        return MinIPOnly(
            grad_checkpoint=args.grad_checkpoint,
            deep_supervision=not args.no_deep_supervision,
        )
    return DSANet(
        num_frames=args.num_frames,
        tf_layers=args.tf_layers,
        tf_heads=args.tf_heads,
        alpha_mode=args.alpha_mode,
        attn_backend=args.attn_backend,
        grad_checkpoint=args.grad_checkpoint,
        deep_supervision=not args.no_deep_supervision,
    )


def load_fold(data_root: Path, fold: int) -> tuple[list[str], list[str]]:
    folds_path = data_root / "folds.json"
    if not folds_path.exists():
        raise FileNotFoundError(f"{folds_path} 不存在，请先运行 data/prepare.py")
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    key = f"fold_{fold}"
    if key not in folds:
        raise KeyError(f"{key} 不在 {folds_path} 里（可选 0~{len(folds)-1}）")
    return folds[key]["train"], folds[key]["val"]


@torch.no_grad()
def validate(model, ds: DSCADataset, device, use_amp: bool,
             tta: bool = True) -> dict:
    """逐例推理（不用 DataLoader，避免增强/批处理干扰），返回聚合 Dice。"""
    was_training = model.training
    model.eval()
    all_dice, all_present = [], []
    for i in range(len(ds)):
        seq, minip, label = ds[i]
        seq = seq.unsqueeze(0).to(device)          # (1,T,1,H,W)
        minip = minip.unsqueeze(0).to(device)      # (1,1,H,W)
        pred = predict_label(model, seq, minip, use_amp=use_amp, tta=tta)[0].cpu().numpy()
        d, p = dice_per_class(pred, label.numpy())
        all_dice.append(d)
        all_present.append(p)
    if was_training:
        model.train()
    return aggregate_dice(all_dice, all_present)


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="DSCA / DSANet 训练")
    ap.add_argument("--data", default=r"E:\Datasets\DSCA\processed")
    ap.add_argument("--size", type=int, default=0,
                    help="⚠️ 只用于校验/诊断，**不会**改变输入尺寸（预处理已定死形状）。"
                         "0=自动从数据推断（推荐）。填了但与数据不符会直接报错 —— "
                         "旋转增强的旋转中心由它算出，错了会静默平移图像。")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--arch", default="dsanet", choices=["dsanet", "minip_only"],
                    help="网络结构。dsanet = 论文主模型（双输入 + 时序模块）；"
                         "minip_only = 论文 Table VI 的 MinIP-only 基线"
                         "（只有 SEB + Decoder，用于判断差距出在基础还是时序模块）")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--iters-per-epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="本机只跑得动 1；论文是 2，用 --accum 2 等效")
    ap.add_argument("--accum", type=int, default=2,
                    help="梯度累积步数，等效 batch = batch_size * accum")
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"],
                    help="论文用 SGD。adam 只作为「SGD 从随机初始化太久才逃出全背景盆地」时的"
                         "退路（此时配 --lr 1e-3），汇报时要说明这是偏离论文的地方")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.99)
    ap.add_argument("--weight-decay", type=float, default=3e-5)
    ap.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True,
                    help="SGD 用不用 Nesterov 动量。nnUNet 的配方是 True；"
                         "关掉后从随机初始化很难逃出「全预测背景」的盆地")
    ap.add_argument("--poly-power", type=float, default=0.9)
    ap.add_argument("--lr-total-epochs", type=int, default=0,
                    help="学习率曲线的总长度（0 = 用 --epochs）。"
                         "跑短周期 probe 时务必设成 500（论文值）："
                         "否则 lr 会在 probe 期间被压缩着衰减（12ep 时只剩 10.7%%），"
                         "Dice 会被严重低估，见 poly_lr 的注释")
    ap.add_argument("--clip-grad", type=float, default=12.0)
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--tf-layers", type=int, default=4)
    ap.add_argument("--tf-heads", type=int, default=8)
    ap.add_argument("--alpha-mode", default="attention",
                    choices=["attention", "spatial_gate"])
    ap.add_argument("--attn-backend", default="sdpa", choices=["sdpa", "manual"])
    ap.add_argument("--norm", default=DEFAULT_NORM, choices=list(NORM_MODES),
                    help="归一化方式。默认 per-image z-score —— 原始数据跨样本像素尺度差 25.5 倍，"
                         "固定 /4095 会让 30%% 的样本近乎全黑（详见 data/dataset.py 注释）")
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--dice-include-bg", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Dice 是否算背景类。默认 False（背景占 94~96%%，算进去会把前景梯度淹掉）")
    ap.add_argument("--class-weights", default="none", choices=["none", "invfreq"],
                    help="CE 类别权重。invfreq = 按训练集频率取倒数（对手写 Dataset 会先扫一遍）")
    ap.add_argument("--focal-gamma", type=float, default=0.0,
                    help=">0 时把 CE 换成 focal loss（0 表示普通 CE）")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-every", type=int, default=10, help="每多少 epoch 验证一次")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--elastic", action="store_true", help="开启弹性形变（会明显变慢）")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-deep-supervision", action="store_true")
    ap.add_argument("--val-tta", action=argparse.BooleanOptionalAction, default=False,
                    help="验证时是否用镜像 TTA（训练中默认关闭以省时间）")
    ap.add_argument("--limit-train", type=int, default=0,
                    help="只用前 N 个训练样本（0=全用）。给「小样本过拟合探针」用："
                         "配 --lr-total-epochs 500 一起用，别让 lr 被提前掐死")
    ap.add_argument("--limit-val", type=int, default=0,
                    help="只用前 N 个验证样本（0=全用）")
    ap.add_argument("--resume", default="",
                    help="从断点续训。给 ckpt 路径，或写 'auto' 读 <out>/last.pth。"
                         "会恢复 模型权重 + 优化器状态 + epoch + best + history，"
                         "lr 曲线按绝对 epoch 继续（不会从头重来）")
    args = ap.parse_args()

    set_seed(args.seed)
    data_root = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    def log(msg: str = "") -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log("⚠️ 没有可用 CUDA，将在 CPU 上训练（会极慢）")

    train_cores, val_cores = load_fold(data_root, args.fold)
    n_train_all, n_val_all = len(train_cores), len(val_cores)
    if args.limit_train:
        train_cores = train_cores[: args.limit_train]
    if args.limit_val:
        val_cores = val_cores[: args.limit_val]

    log("=" * 78)
    log(f"  DSCA / DSANet 训练   fold={args.fold}   out={out_dir}")
    log("=" * 78)
    log(f"  设备            : {device}"
        + (f"  {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    log(f"  训练 / 验证     : {len(train_cores)} / {len(val_cores)} 例")
    if len(train_cores) != n_train_all or len(val_cores) != n_val_all:
        log(f"  ⚠️ 子集模式     : 训练 {len(train_cores)}/{n_train_all}，"
            f"验证 {len(val_cores)}/{n_val_all}（--limit-train/--limit-val）"
            f" —— 结果不可与全量实验直接比较")
    log(f"  数据目录        : {data_root}")
    log(f"  epoch × iter    : {args.epochs} × {args.iters_per_epoch}")
    log(f"  batch × accum   : {args.batch_size} × {args.accum} = {args.batch_size*args.accum}")
    log(f"  优化器          : {args.optimizer.upper()} lr={args.lr} "
        f"wd={args.weight_decay}"
        + (f" momentum={args.momentum} nesterov={args.nesterov}"
           if args.optimizer == "sgd" else "  ⚠️ 偏离论文（论文是 SGD）"))
    _lrtot = args.lr_total_epochs or args.epochs
    log(f"  衰减            : polynomial power={args.poly_power}  "
        f"lr 曲线总长 {_lrtot} epoch"
        + (f"  ⚠️ 本实验只跑 {args.epochs} epoch（=论文前 "
           f"{100.0 * args.epochs / _lrtot:.1f}% 的训练阶段），"
           f"lr 不会被提前掐死" if _lrtot != args.epochs else ""))
    log(f"  AMP / 梯度检查点: {args.amp} / {args.grad_checkpoint}")
    log(f"  alpha_mode      : {args.alpha_mode}")
    log(f"  架构            : {args.arch}")
    log(f"  归一化          : {args.norm}")
    log(f"  损失            : CE×{args.ce_weight} + Dice×{args.dice_weight}"
        f"（Dice {'含' if args.dice_include_bg else '不含'}背景）"
        f" + 辅助×{args.aux_weight}   权重={args.class_weights}  focal γ={args.focal_gamma}")
    log("")

    train_ds = DSCADataset(data_root, "train", cores=train_cores,
                           num_frames=args.num_frames, augment=True,
                           elastic=args.elastic, seed=args.seed,
                           size=(args.size or None), norm=args.norm)
    val_ds = DSCADataset(data_root, "train", cores=val_cores,
                         num_frames=args.num_frames, augment=False,
                         size=(args.size or None), norm=args.norm)
    log(f"数据尺寸 self.size = {train_ds.size}（由数据推断；只影响增强的旋转中心）")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=args.num_workers > 0)

    model = build_model(args).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"  模型参数量      : {n_par:,} ({n_par/1e6:.2f} M)")
    log("")

    # CE 类别权重（可选）。只读 npz 里的 label，不会把 seq 也解出来。
    class_weights = None
    if args.class_weights == "invfreq":
        counts = np.zeros(3, dtype=np.int64)
        for c in train_cores:
            lab = np.load(data_root / "train" / f"{c}.npz")["label"]
            vals, cnts = np.unique(lab, return_counts=True)
            for v, n in zip(vals.tolist(), cnts.tolist()):
                if v < 3:
                    counts[v] += n
        class_weights = inverse_frequency_weights(counts, 3)
        frac = counts / counts.sum()
        log(f"  CE 类别权重     : {[round(float(x), 3) for x in class_weights]}"
            f"   训练集像素占比 {[round(float(x), 4) for x in frac]}")
        log("")

    criterion = DeepSupervisionLoss(
        n_classes=3,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        aux_weight=args.aux_weight,
        include_background_in_dice=args.dice_include_bg,
        class_weights=class_weights,
        focal_gamma=args.focal_gamma,
    ).to(device)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)

    best = -1.0
    history: list[dict] = []
    start_epoch = 0

    # ---- 断点续训（2026-09-26 加：之前没有 resume，f3 卡死只能从 epoch 0 重跑）----
    if args.resume:
        ckpt_path = (out_dir / "last.pth") if args.resume == "auto" else Path(args.resume)
        if not ckpt_path.exists():
            log(f"  ⚠️ --resume 目标不存在：{ckpt_path} → 从 epoch 0 重新开始")
        else:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            if ck.get("optimizer"):
                optimizer.load_state_dict(ck["optimizer"])
                log("  ✅ 已恢复优化器状态（动量 / 自适应矩）")
            else:
                log("  ⚠️ ckpt 里没有优化器状态（旧格式），动量会重置 → 短期抖动")
            start_epoch = int(ck.get("epoch", 0))
            history = list(ck.get("history", []))
            _d = ck.get("dice")
            _from_dice = float(_d["mean_foreground"]) if isinstance(_d, dict) else float(_d or -1.0)
            best = float(ck.get("best", _from_dice))
            log(f"  ✅ 续训：{ckpt_path.name}  epoch {start_epoch} 起，"
                f"best 前景 Dice {best:.4f}，已载入 {len(history)} 条历史")
            if start_epoch >= args.epochs:
                log(f"  ⚠️ ckpt 已到 epoch {start_epoch} ≥ --epochs {args.epochs}，"
                    f"没有要继续的训练了（把 --epochs 调大）")
                return 0

    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        lr = poly_lr(epoch, args.lr_total_epochs or args.epochs,
                     args.lr, args.poly_power)
        for g in optimizer.param_groups:
            g["lr"] = lr

        model.train()
        it = iter(loader)
        losses: list[float] = []
        t_ep = time.time()

        for step in range(args.iters_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(args.accum):
                try:
                    seq, minip, label = next(it)
                except StopIteration:
                    it = iter(loader)
                    seq, minip, label = next(it)
                seq = seq.to(device, non_blocking=True)
                minip = minip.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=args.amp and device.type == "cuda"):
                    main, aux = model(seq, minip)
                    loss, parts = criterion(main, aux, label)
                    loss = loss / args.accum
                loss.backward()
                step_loss += parts["loss_total"]

            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            losses.append(step_loss / args.accum)

            if (step + 1) % max(1, args.iters_per_epoch // 5) == 0:
                recent = sum(losses[-max(1, len(losses)//5):]) / max(1, len(losses)//5)
                log(f"  [fold{args.fold}] epoch {epoch+1}/{args.epochs} "
                    f"step {step+1}/{args.iters_per_epoch}  "
                    f"lr {lr:.5f}  loss {recent:.4f}  ({time.time()-t_ep:.0f}s)")

        mean_loss = sum(losses) / max(1, len(losses))
        rec = {"epoch": epoch + 1, "loss": mean_loss, "lr": lr}

        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            agg = validate(model, val_ds, device, args.amp, tta=args.val_tta)
            rec["val_dice"] = agg
            log(f"  >>> epoch {epoch+1} 完成  mean_loss {mean_loss:.4f}  "
                f"val Dice: {format_dice(agg)}")
            if agg.get("mean_foreground", -1) > best:
                best = agg["mean_foreground"]
                torch.save({"model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),      # resume 用
                            "args": vars(args), "epoch": epoch + 1,
                            "dice": agg, "best": best,
                            "history": history}, out_dir / "best.pth")
                log(f"      ↳ 新的最佳前景 Dice {best:.4f}，已保存 best.pth")
        else:
            log(f"  >>> epoch {epoch+1} 完成  mean_loss {mean_loss:.4f}")

        history.append(rec)
        (out_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),                  # resume 用
                "args": vars(args), "epoch": args.epochs,
                "dice": best,                  # float（保持旧语义；best.pth 里是 dict）
                "best": best, "history": history}, out_dir / "last.pth")

    log("")
    log("=" * 78)
    log(f"  训练结束，总用时 {(time.time()-t_start)/3600:.2f} 小时")
    log(f"  最佳前景 Dice : {best:.4f}")
    log(f"  权重          : {out_dir/'best.pth'}  /  {out_dir/'last.pth'}")
    log(f"  训练曲线数据  : {out_dir/'history.json'}")
    log("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
