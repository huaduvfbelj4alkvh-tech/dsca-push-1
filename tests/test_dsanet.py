"""L3 DSANet 端到端验收测试。

判据：
    D1 小尺寸端到端 shape：(B,T,1,H,W)+(B,1,H,W) -> (B,3,H,W)
    D2 深监督辅助输出的数量与形状
    D3 前向 + 反向，梯度健全
    D4 use_injection 开关确实改变输出（证明 TEB->SEB 的逐层注入生效）
    D5 参数量统计（分模块）
    D6 🔴 真实尺寸 512×512 端到端显存实测（batch=1 + AMP）—— 这条决定 train.py 能不能用
    D7 非法输入报错

运行：E:\\Anaconda\\envs\\dsca\\python.exe F:\\Projects\\DSCA\\tests\\test_dsanet.py
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from models.dsanet import DSANet  # noqa: E402

torch.manual_seed(0)
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


SMALL_CHS = (8, 16, 32, 64, 128)
B, T, H, W = 1, 8, 64, 64


def small_model(**kw) -> DSANet:
    return DSANet(chs=SMALL_CHS, tf_heads=4, num_frames=T, **kw)


print("=" * 74)
print("D1/D2  小尺寸端到端")
print("=" * 74)
model = small_model().eval()
seq = torch.randn(B, T, 1, H, W)
minip = torch.randn(B, 1, H, W)
with torch.no_grad():
    main, aux = model(seq, minip)
check("D1  主输出形状 (1,3,64,64)", main.shape == (B, 3, H, W), f"{tuple(main.shape)}")
check("D2  深监督辅助输出 3 个", len(aux) == 3, f"得到 {len(aux)} 个")
print("     辅助输出形状:", [tuple(a.shape) for a in aux])

no_ds = small_model(deep_supervision=False).eval()
with torch.no_grad():
    _, aux0 = no_ds(seq, minip)
check("D2b 关闭深监督后无辅助输出", len(aux0) == 0, f"得到 {len(aux0)} 个")

print()
print("=" * 74)
print("D3  前向 + 反向")
print("=" * 74)
model = small_model().train()
seq = torch.randn(B, T, 1, H, W, requires_grad=True)
minip = torch.randn(B, 1, H, W, requires_grad=True)
main, aux = model(seq, minip)
loss = main.square().mean() + sum(a.square().mean() for a in aux)
loss.backward()
n_with = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
n_tot = sum(1 for _ in model.parameters())
check("D3a 输入梯度无 NaN",
      not torch.isnan(seq.grad).any() and not torch.isnan(minip.grad).any(),
      f"seq grad.norm={seq.grad.norm():.3f}")
check("D3b 全部参数收到梯度", n_with == n_tot, f"{n_with}/{n_tot}")

# 重点检查"深处"的参数确实拿到了梯度（注入链路、TF、STF 都属于容易断梯度的地方）
grad_report = {}
for name, p in model.named_parameters():
    if p.grad is None:
        grad_report[name] = 0.0
    else:
        grad_report[name] = float(p.grad.abs().sum())
for key in ("tf.pos_embed", "stf."):
    hits = [v for k, v in grad_report.items() if k.startswith(key)]
    ok = bool(hits) and all(v > 0 for v in hits)
    check(f"D3c {key}* 的梯度全部非零", ok, f"{len(hits)} 个参数")

print()
print("=" * 74)
print("D4  TEB -> SEB 逐层注入开关（Fig.3 的虚线，正文没写）")
print("=" * 74)
torch.manual_seed(7)
seq = torch.randn(B, T, 1, H, W)
minip = torch.randn(B, 1, H, W)
into = small_model(use_injection=True).eval()
nout = small_model(use_injection=False).eval()
nout.load_state_dict(into.state_dict(), strict=False)
with torch.no_grad():
    a, _ = into(seq, minip)
    b, _ = nout(seq, minip)
diff = (a - b).abs().max().item()
check("D4  开关确实改变输出", diff > 1e-6, f"max|diff| = {diff:.4f}")

print()
print("=" * 74)
print("D5  参数量（论文配置 chs=32..512, T=8）")
print("=" * 74)
full = DSANet(num_frames=8)
total = n_params(full)
print(f"  DSANet 总参数量      : {total:,}  ({total/1e6:.2f} M)")
for name, mod in (("TEB", full.teb), ("SEB", full.seb), ("TF", full.tf),
                  ("STF", full.stf), ("Decoder", full.decoder)):
    p = n_params(mod)
    print(f"    {name:<8}: {p:>10,}  ({p/total*100:5.1f}%)")
check("D5  参数量已统计", total > 0, f"{total/1e6:.2f} M")

print()
print("=" * 74)
print("D6  🔴 真实尺寸显存：batch=1, T=8, 512×512")
print("=" * 74)
if not torch.cuda.is_available():
    check("D6  显存", False, "CUDA 不可用")
else:
    dev = torch.device("cuda")
    free0, _ = torch.cuda.mem_get_info()
    avail = free0 / 1024 ** 2
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
    print(f"  起始空闲显存   : {avail:.0f} MiB\n")

    def run(amp: bool, gc: bool, bs: int = 1) -> tuple[bool, float, str]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m = DSANet(num_frames=8, grad_checkpoint=gc).to(dev).train()
        s = torch.randn(bs, 8, 1, 512, 512, device=dev)
        p = torch.randn(bs, 1, 512, 512, device=dev)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                main, aux = m(s, p)
                loss = main.square().mean() + 0.5 * sum(a.square().mean() for a in aux)
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
            return True, peak, ""
        except torch.cuda.OutOfMemoryError as e:
            return False, float("nan"), str(e)[:70]
        finally:
            del m, s, p
            torch.cuda.empty_cache()

    best: tuple[str, float] | None = None
    for tag, amp, gc in (("fp32", False, False),
                         ("AMP(bf16)", True, False),
                         ("AMP(bf16) + 梯度检查点", True, True)):
        ok, peak, err = run(amp, gc)
        if ok:
            verdict = "✅ 真显存内" if peak < avail else "⚠️ 分页到内存（会极慢）"
            print(f"  batch=1 {tag:<22} 峰值 {peak:7.0f} MiB   {verdict}")
            if peak < avail and best is None:
                best = (tag, peak)
        else:
            print(f"  batch=1 {tag:<22} 失败：{err}")

    check("D6  存在能装进真显存的配置", best is not None,
          f"推荐 [{best[0]}] 峰值 {best[1]:.0f} MiB" if best else "全部超限，需再优化")

print()
print("=" * 74)
print("D7  非法输入")
print("=" * 74)
m = small_model().eval()
with torch.no_grad():
    try:
        m(torch.randn(B, 4, 1, H, W), minip)
        check("D7a 帧数不符报错", False, "未抛出")
    except ValueError:
        check("D7a 帧数不符报错", True, "ValueError")
    try:
        m(seq, torch.randn(B, 1, H + 2, W))
        check("D7b MinIP 尺寸不符报错", False, "未抛出")
    except ValueError:
        check("D7b MinIP 尺寸不符报错", True, "ValueError")
    try:
        m(torch.randn(B, T, H, W), minip)               # 4 维，少一维
        check("D7c seq 维度不足报错", False, "未抛出")
    except ValueError:
        check("D7c seq 维度不足报错", True, "ValueError")

print()
print("=" * 74)
n_pass = sum(1 for _, ok, _ in RESULTS if ok)
print(f"  {n_pass} / {len(RESULTS)} 通过")
for n, ok, d in RESULTS:
    if not ok:
        print(f"  FAILED -> {n}  {d}")
print("  L3 结论:", "✅ 通过" if n_pass == len(RESULTS) else "❌ 有失败项")
sys.exit(0 if n_pass == len(RESULTS) else 1)
