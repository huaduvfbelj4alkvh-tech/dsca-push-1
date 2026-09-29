"""L1 TemporalFormer 验收测试 —— 按论文维度链做手算对拍。

为什么要有这个文件:
    TF 的维度链里有 4 次 reshape/rearrange 在同一条链上往返。
    一旦轴序写错（例如把 (h w t) 写成 (t h w)），**代码不会报错**——
    shape 全对，只是帧与帧、像素与像素的对应关系错位，最后表现为
    "能训但精度莫名偏低"。这种 bug 无法靠肉眼审代码发现，只能靠对拍。

判据（全 PASS 才算 L1 完成）:
    T1 轴序对拍        —— 手工索引验证 (B,t,c,h,w) <-> (B,h,w,t) 的映射，且两条路径一致
    T2 空域往返恒等    —— (b t)(h w)c 与 b(h w t)c 的往返必须逐元素相等
    T3 MHSA 数值对拍   —— 与逐头逐位置手算的朴素实现比，atol 1e-5
    T4 端到端 shape    —— (B*T,c,h,w) -> (B,c,h,w)，含非方阵 h != w 的情形
    T5 前向 + 反向     —— 梯度非 NaN，pos_embed 确实收到梯度
    T6 确定性          —— eval 模式两次前向逐元素相同
    T7 4 种 T 维下采样模式都能跑通且输出形状一致
    T8 参数量 / 峰值显存（论文设置 batch=1, T=8, c=512, 32x32 瓶颈）

运行:
    E:\\Anaconda\\envs\\dsca\\python.exe F:\\Projects\\DSCA\\tests\\test_temporalformer.py
"""

from __future__ import annotations

import pathlib
import sys

import torch
from einops import rearrange

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from models.temporalformer import MHSA, TemporalFormer, TemporalDownsample  # noqa: E402

torch.manual_seed(0)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ============================================================================
section("T1  轴序对拍：手工索引验证 (B,t,c,h,w) <-> (B,h,w,t) 的映射")

B, T, C, H, W = 1, 4, 2, 2, 3
x = torch.arange(B * T * C * H * W, dtype=torch.float32).reshape(B * T, C, H, W)

# 路径 A：论文写法，先并成 (B, h*w*t, c)
y_flat = rearrange(x, '(b t) c h w -> b (h w t) c', b=B, t=T)
# 路径 B：直接并成 (B*h*w, t, c)（这是论文 (Bhw),T,c 那一步）
y_bhw = rearrange(x, '(b t) c h w -> (b h w) t c', b=B, t=T)
# 路径 A 再变回路径 B 的形状
y_flat_back = rearrange(y_flat, 'b (h w t) c -> (b h w) t c', h=H, w=W, t=T)

check("T1a (B,h,w,t) 与 (Bhw,T) 两条路径完全一致",
      torch.equal(y_flat_back, y_bhw),
      "reshape 链自洽")

# 手工逐元素验证 (B, h*w*t, c) 的展平顺序：h 最外层、w 中间、t 最内层
mismatch = 0
for h in range(H):
    for w in range(W):
        for t in range(T):
            k = h * (W * T) + w * T + t
            if not torch.equal(y_flat[0, k], x[t, :, h, w]):
                mismatch += 1
check("T1b 展平索引 k = h*(W*T) + w*T + t 逐元素成立",
      mismatch == 0, f"错位 {mismatch} 处 / 共 {H*W*T} 处")

# 反向：从 (B,h,w,t) 还原回 (B*T,c,h,w)
x_back = rearrange(y_flat, 'b (h w t) c -> (b t) c h w', b=B, t=T, h=H, w=W)
check("T1c (B,h,w,t) -> (B*T,c,h,w) 往返无损",
      torch.equal(x_back, x), "往返完全还原")


# ============================================================================
section("T2  空域往返恒等：(b t)(h w)c <-> b(h w t)c")

zs = rearrange(y_flat, 'b (h w t) c -> (b t) (h w) c', h=H, w=W, t=T)
z_back = rearrange(zs, '(b t) (h w) c -> b (h w t) c', h=H, w=W, t=T)
check("T2a 空域分支往返恒等", torch.equal(z_back, y_flat), "逐元素相等")

# 严格检查：zs[t] 的第 p 行应等于 y_flat[0, h*W*T + w*T + t]（p = h*W + w）
strict_mismatch = 0
for t in range(T):
    for h in range(H):
        for w in range(W):
            p = h * W + w
            if not torch.equal(zs[t, p], y_flat[0, h * W * T + w * T + t]):
                strict_mismatch += 1
check("T2b (b t) 轴上第 t 个样本 = 各空间位置在该帧的特征",
      strict_mismatch == 0, f"错位 {strict_mismatch} 处 / 共 {T*H*W} 处")


# ============================================================================
section("T3  MHSA 数值对拍：与逐头逐位置手算的朴素实现比")

torch.manual_seed(1)
dim, heads, L = 8, 2, 5
mhsa = MHSA(dim, heads).eval()
x0 = torch.randn(L, dim)
with torch.no_grad():
    got = mhsa(x0.unsqueeze(0)).squeeze(0)          # (L, dim)


def naive_mhsa_single(x0: torch.Tensor, m: MHSA) -> torch.Tensor:
    """逐位置、逐头、逐维度手算注意力。完全不使用 bmm/matmul 的批量写法。"""
    length, c = x0.shape
    hn, d = m.num_heads, m.head_dim
    qkv = x0 @ m.qkv.weight.T + m.qkv.bias           # (L, 3C)
    q, k, v = qkv.split(c, dim=-1)

    outs = []
    for i in range(length):
        heads_out = []
        for hh in range(hn):
            qs = q[i, hh * d:(hh + 1) * d]
            logits = []
            for j in range(length):
                ks = k[j, hh * d:(hh + 1) * d]
                s = 0.0
                for dd in range(d):
                    s += float(qs[dd]) * float(ks[dd])
                logits.append(s * m.scale)
            logits_t = torch.tensor(logits)
            w = torch.softmax(logits_t, dim=0)
            acc = torch.zeros(d)
            for j in range(length):
                acc = acc + w[j] * v[j, hh * d:(hh + 1) * d]
            heads_out.append(acc)
        outs.append(torch.cat(heads_out))
    out = torch.stack(outs)                           # (L, C)
    return out @ m.proj.weight.T + m.proj.bias


with torch.no_grad():
    ref = naive_mhsa_single(x0, mhsa)

diff = (got - ref).abs().max().item()
check("T3  MHSA 与手算朴素实现一致 (atol 1e-5)", diff < 1e-5, f"max|diff| = {diff:.3e}")

# T3b: 两种注意力后端（SDPA / 手工）必须数值一致 —— 否则"省显存"就没意义了
mhsa_manual = MHSA(dim, heads, attn_backend="manual").eval()
mhsa_manual.load_state_dict(mhsa.state_dict())
with torch.no_grad():
    got_manual = mhsa_manual(x0.unsqueeze(0)).squeeze(0)
d2 = (got - got_manual).abs().max().item()
check("T3b sdpa 与 manual 后端数值一致", d2 < 1e-6, f"max|diff| = {d2:.3e}")


# ============================================================================
section("T4  端到端 shape：含非方阵 h != w 与 batch > 1")

tf = TemporalFormer(dim=32, num_layers=2, num_heads=4, num_frames=4).eval()

for (bb, tt, cc, hh, ww) in [(2, 4, 32, 8, 8), (1, 4, 32, 5, 7), (3, 4, 32, 6, 6)]:
    inp = torch.randn(bb * tt, cc, hh, ww)
    with torch.no_grad():
        out = tf(inp)
    ok = out.shape == (bb, cc, hh, ww)
    check(f"T4  (B={bb},T={tt},c={cc},{hh}x{ww}) -> {tuple(out.shape)}", ok,
          "shape 正确" if ok else f"期望 {(bb, cc, hh, ww)}")

try:
    tf(torch.randn(5, 32, 8, 8))                      # 5 不是 num_frames=4 的倍数
    check("T4b 非法 batch 应当报错", False, "没有抛异常")
except ValueError as e:
    check("T4b 非法 batch 应当报错", True, "已抛出 ValueError")


# ============================================================================
section("T5  前向 + 反向：梯度健全性")

tf = TemporalFormer(dim=32, num_layers=2, num_heads=4, num_frames=4).train()
inp = torch.randn(8, 32, 8, 8, requires_grad=True)
out = tf(inp)
loss = out.square().mean()
loss.backward()

nan_in = torch.isnan(inp.grad).any().item()
nan_pos = torch.isnan(tf.pos_embed.grad).any().item() if tf.pos_embed.grad is not None else True
grad_norm = float(tf.pos_embed.grad.norm()) if tf.pos_embed.grad is not None else 0.0
n_with_grad = sum(1 for p in tf.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
n_total = sum(1 for _ in tf.parameters())

check("T5a 输入梯度无 NaN", not nan_in, f"grad.norm = {inp.grad.norm():.4f}")
check("T5b pos_embed 收到非零梯度", (not nan_pos) and grad_norm > 0, f"|grad| = {grad_norm:.3e}")
check("T5c 所有参数都收到梯度", n_with_grad == n_total, f"{n_with_grad}/{n_total}")


# ============================================================================
section("T6  确定性：eval 模式两次前向必须逐元素相同")

tf = TemporalFormer(dim=32, num_layers=2, num_heads=4, num_frames=4).eval()
inp = torch.randn(8, 32, 8, 8)
with torch.no_grad():
    o1, o2 = tf(inp), tf(inp)
check("T6  eval 模式两次前向一致", torch.equal(o1, o2), "逐元素相等")


# ============================================================================
section("T7  4 种 T 维下采样模式")

for mode in TemporalDownsample.VALID_MODES:
    try:
        tfd = TemporalFormer(dim=32, num_layers=1, num_heads=4, num_frames=4,
                             temporal_down=mode).eval()
        with torch.no_grad():
            o = tfd(torch.randn(8, 32, 8, 8))
        ok = o.shape == (2, 32, 8, 8)
        check(f"T7  temporal_down={mode!r}", ok, f"out = {tuple(o.shape)}")
    except Exception as e:                             # noqa: BLE001
        check(f"T7  temporal_down={mode!r}", False, f"{type(e).__name__}: {e}")


# ============================================================================
section("T8  论文设置下的参数量与显存（batch=1, T=8, c=512, 32x32 瓶颈）")


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


big = TemporalFormer(dim=512, num_layers=4, num_heads=8, num_frames=8)
n_tf = count_params(big)
print(f"  TF 总参数量            : {n_tf:,}  ({n_tf/1e6:.2f} M)")
print(f"  其中 pos_embed         : {big.pos_embed.numel():,}")
print(f"  其中 temporal_down     : {count_params(big.temporal_down):,}")
for i, layer in enumerate(big.layers):
    print(f"    第 {i} 层              : {count_params(layer):,}")

if torch.cuda.is_available():
    dev = torch.device("cuda")
    free0, _ = torch.cuda.mem_get_info()
    print(f"\n  GPU                    : {torch.cuda.get_device_name(0)}")
    print(f"  起始空闲显存           : {free0/1024**2:.0f} MiB\n")

    peaks: dict[str, float] = {}
    for backend in ("manual", "sdpa"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m = TemporalFormer(dim=512, num_layers=4, num_heads=8, num_frames=8,
                           attn_backend=backend).to(dev).train()
        inp = torch.randn(8, 512, 32, 32, device=dev)
        out = m(inp)
        out.square().mean().backward()
        peaks[backend] = torch.cuda.max_memory_allocated() / 1024 ** 2
        tag = "✅ 真显存内" if peaks[backend] < free0 / 1024 ** 2 else "⚠️ 分页到内存"
        print(f"  attn_backend={backend:<7} 峰值 {peaks[backend]:7.0f} MiB  "
              f"输出 {tuple(out.shape)}  {tag}")
        del m, inp, out
        torch.cuda.empty_cache()

    print(f"\n  SDPA 相对 manual 节省 : {100 * (1 - peaks['sdpa'] / peaks['manual']):.1f}%")
    check("T8  TF 在 batch=1 下峰值不超过起始空闲显存",
          peaks["sdpa"] < free0 / 1024 ** 2,
          f"sdpa {peaks['sdpa']:.0f} MiB vs 可用 {free0/1024**2:.0f} MiB")
else:
    print("  未检测到 CUDA，跳过显存测试")
    check("T8  显存测试", False, "CUDA 不可用")


# ============================================================================
section("汇总")
n_pass = sum(1 for _, ok, _ in RESULTS if ok)
print(f"\n  {n_pass} / {len(RESULTS)} 通过")
for name, ok, detail in RESULTS:
    if not ok:
        print(f"  FAILED -> {name}  {detail}")
print("\n  L1 结论:", "✅ 全部通过，TF 维度链可信" if n_pass == len(RESULTS) else "❌ 存在失败项，先修再往下")
sys.exit(0 if n_pass == len(RESULTS) else 1)
