"""L2 STF 验收测试。

判据：
    S1 形状：(B,c,h,w) × 2 -> (B,2c,h,w)
    S2 两种 alpha_mode 都跑通，且输出确实不同（证明开关真的在起作用，而不是写死了一种）
    S3 非方阵 h≠w
    S4 share_qkv 开关改变参数量
    S5 前向 + 反向，全部参数收到梯度
    S6 非法输入必须报错

运行：E:\\Anaconda\\envs\\dsca\\python.exe F:\\Projects\\DSCA\\tests\\test_stf.py
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from models.stf import SpatialTemporalFusion  # noqa: E402

torch.manual_seed(0)
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


print("=" * 74)
print("S1 形状")
print("=" * 74)
stf = SpatialTemporalFusion(dim=32).eval()
f_m = torch.randn(2, 32, 16, 16)
f_s = torch.randn(2, 32, 16, 16)
with torch.no_grad():
    out = stf(f_m, f_s)
check("S1  (B=2,c=32,16x16) -> (2,64,16,16)", out.shape == (2, 64, 16, 16), f"{tuple(out.shape)}")

with torch.no_grad():
    out_ns = stf(torch.randn(1, 32, 8, 12), torch.randn(1, 32, 8, 12))
check("S3  非方阵 (8x12) 也能跑", out_ns.shape == (1, 64, 8, 12), f"{tuple(out_ns.shape)}")

print()
print("=" * 74)
print("S2  alpha_mode 开关")
print("=" * 74)
torch.manual_seed(123)
f_m = torch.randn(2, 32, 16, 16)
f_s = torch.randn(2, 32, 16, 16)
outs = {}
for mode in ("attention", "spatial_gate"):
    m = SpatialTemporalFusion(dim=32, alpha_mode=mode).eval()
    with torch.no_grad():
        outs[mode] = m(f_m, f_s)
    check(f"S2  alpha_mode={mode!r} 跑通", outs[mode].shape == (2, 64, 16, 16),
          f"{tuple(outs[mode].shape)}")

same = torch.allclose(outs["attention"], outs["spatial_gate"], atol=1e-6)
check("S2b 两种读法输出确实不同", not same,
      "相同 -> 说明开关没生效" if same else "已确认是两条不同计算路径")

n_att = n_params(SpatialTemporalFusion(dim=32, alpha_mode="attention"))
n_gate = n_params(SpatialTemporalFusion(dim=32, alpha_mode="spatial_gate"))
check("S2c 两种读法参数量相同（靠消融数字才能判对错）", n_att == n_gate,
      f"attention {n_att:,} vs spatial_gate {n_gate:,}")

print()
print("=" * 74)
print("S4  share_qkv 开关")
print("=" * 74)
n_sep = n_params(SpatialTemporalFusion(dim=32, share_qkv=False))
n_sha = n_params(SpatialTemporalFusion(dim=32, share_qkv=True))
check("S4  共享 Q/K/V 参数量减半", n_sha * 2 == n_sep, f"独立 {n_sep:,} -> 共享 {n_sha:,}")

print()
print("=" * 74)
print("S5  前向 + 反向")
print("=" * 74)
m = SpatialTemporalFusion(dim=32).train()
f_m = torch.randn(2, 32, 16, 16, requires_grad=True)
f_s = torch.randn(2, 32, 16, 16, requires_grad=True)
m(f_m, f_s).square().mean().backward()
n_with = sum(1 for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
n_tot = sum(1 for _ in m.parameters())
check("S5a 两路输入都收到梯度",
      f_m.grad is not None and f_s.grad is not None
      and not torch.isnan(f_m.grad).any() and not torch.isnan(f_s.grad).any())
check("S5b 全部参数收到梯度", n_with == n_tot, f"{n_with}/{n_tot}")

print()
print("=" * 74)
print("S6  非法输入")
print("=" * 74)
try:
    SpatialTemporalFusion(dim=32, alpha_mode="nope")
    check("S6a 非法 alpha_mode 报错", False, "未抛出")
except ValueError:
    check("S6a 非法 alpha_mode 报错", True, "ValueError")
try:
    SpatialTemporalFusion(dim=32)(torch.randn(1, 32, 8, 8), torch.randn(1, 32, 9, 9))
    check("S6b F_m/F_s 形状不一致报错", False, "未抛出")
except ValueError:
    check("S6b F_m/F_s 形状不一致报错", True, "ValueError")

print()
print("=" * 74)
n_pass = sum(1 for _, ok, _ in RESULTS if ok)
print(f"  {n_pass} / {len(RESULTS)} 通过")
for n, ok, d in RESULTS:
    if not ok:
        print(f"  FAILED -> {n}  {d}")
print("  L2 结论:", "✅ 通过" if n_pass == len(RESULTS) else "❌ 有失败项")
sys.exit(0 if n_pass == len(RESULTS) else 1)
