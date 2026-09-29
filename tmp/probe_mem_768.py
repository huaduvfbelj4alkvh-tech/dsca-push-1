"""768 分辨率显存探针：本机 3060 能不能碰 768？（只跑 1 次 fwd+bwd，几十秒）

为什么需要
----------
路线 B（整体上 768）的显存约是 512 的 **2.25 倍**（像素数）。本机 3060 Laptop 可用只有
~5.0 GiB，而 512 用 AMP+梯度检查点已经吃到 **2574 MiB** ⇒ 768 预计 ~5.8 GiB，**大概率不够**。
在集群排队前先量一下，能省一次白跑。

⚠️ 关于"够不够"的判据（踩过的坑）
  * **"不报 OOM" ≠ "装进显存"**：WDDM 会静默把显存溢出到内存，慢几十倍但**不报错**。
    所以既看 `torch.cuda.max_memory_allocated()`，也要看**单 iter 墙钟时间**。
  * `nvidia-smi memory.used`（含上下文/缓存）≠ `max_memory_allocated()`（只算张量），
    两者会差 2 倍以上，别混用。判断"能不能跑"用前者更保守。

用法
----
    E:\\Anaconda\\envs\\dsca\\python.exe tmp/probe_mem_768.py            # 默认 512 768
    E:\\Anaconda\\envs\\dsca\\python.exe tmp/probe_mem_768.py --sizes 512,640,768
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.dsanet import DSANet          # noqa: E402
from models.losses import DeepSupervisionLoss  # noqa: E402


def probe(size: int, frames: int, grad_ckpt: bool, amp: bool) -> dict:
    dev = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()

    model = DSANet(num_frames=frames, grad_checkpoint=grad_ckpt).to(dev).train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    crit = DeepSupervisionLoss(n_classes=3).to(dev)

    seq = torch.randn(1, frames, 1, size, size, device=dev)
    minip = torch.randn(1, 1, size, size, device=dev)
    label = torch.randint(0, 3, (1, size, size), device=dev)

    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        main, aux = model(seq, minip)
        loss, _ = crit(main, aux, label)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
    opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    peak = torch.cuda.max_memory_allocated()
    free_after, _ = torch.cuda.mem_get_info()
    del model, opt, seq, minip, label, loss, main, aux
    torch.cuda.empty_cache()

    return {"size": size, "grad_ckpt": grad_ckpt, "amp": amp,
            "peak_MiB": peak / 2 ** 20, "total_MiB": total / 2 ** 20,
            "free_before_MiB": free_before / 2 ** 20,
            "free_after_MiB": free_after / 2 ** 20,
            "s_per_iter": dt}


def main() -> int:
    ap = argparse.ArgumentParser(description="DSANet 显存/速度探针")
    ap.add_argument("--sizes", default="512,768")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction,
                    default=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA 不可用 —— 这通常意味着装了 CPU 版 torch")
        return 2

    name = torch.cuda.get_device_name(0)
    _, total = torch.cuda.mem_get_info()
    print("=" * 78)
    print(f"DSANet 探针  设备 {name}  总量 {total/2**20:.0f} MiB")
    print(f"  配置 AMP={args.amp}  梯度检查点={args.grad_checkpoint}  T={args.frames}"
          f"  batch=1")
    print("=" * 78)
    print(f"{'size':>6}{'峰值(MiB)':>12}{'占总量':>9}{'剩余(MiB)':>12}{'s/iter':>10}{'判定':>12}")

    for size in [int(s) for s in args.sizes.split(",") if s.strip()]:
        try:
            r = probe(size, args.frames, args.grad_checkpoint, args.amp)
        except torch.cuda.OutOfMemoryError:
            print(f"{size:>6}{'> 总量':>12}{'-':>9}{'-':>12}{'-':>10}{'OOM':>12}")
            torch.cuda.empty_cache()
            continue
        frac = r["peak_MiB"] / r["total_MiB"]
        verdict = "OK" if frac < 0.90 else ("吃紧" if frac < 0.97 else "危险")
        print(f"{size:>6}{r['peak_MiB']:>12.0f}{frac*100:>8.1f}%"
              f"{r['free_after_MiB']:>12.0f}{r['s_per_iter']:>10.1f}{verdict:>12}")

    print()
    print("判读规则")
    print("  * 峰值/总量 < 90% 才算真装得下；90~97% 会被系统缓存挤到溢出（WDDM）。")
    print("  * ⚠️ '不报 OOM' ≠ '装进显存'：注意看 s/iter 有没有莫名放大 10 倍以上。")
    print("  * 判能不能跑用 max_memory_allocated 会偏乐观；要更保守就看 nvidia-smi 的 used。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
