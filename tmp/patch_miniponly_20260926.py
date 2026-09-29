"""给 DSCA 项目加 MinIP-only 基线支持（2026-09-26）。

改 4 个文件：
  1. models/dsanet.py   —— 末尾追加 MinIPOnly 类
  2. models/__init__.py —— 导出 MinIPOnly
  3. train.py           —— import / --arch / build_model / header 打印
  4. evaluate.py        —— import / build_from_ckpt 按 arch 分支

设计要点：
  * 幂等 —— 重复执行不会重复插入；
  * 锚点找不到 / 命中多行 —— 直接报错退出，绝不静默改错；
  * 备份在 *.bak_20260926_miniponly。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def patch_line(rel: str, needle: str, transform) -> None:
    """对**唯一**含 needle 的行做 transform。幂等由 transform 自己保证。"""
    p = ROOT / rel
    lines = p.read_text(encoding="utf-8").split("\n")
    hits = [i for i, ln in enumerate(lines) if needle in ln]
    if len(hits) != 1:
        raise SystemExit(f"  [失败] {rel}: '{needle}' 命中 {len(hits)} 行（应为 1）")
    i = hits[0]
    new = transform(lines[i])
    if new == lines[i]:
        print(f"  [跳过] {rel} : 已是目标状态")
        return
    lines[i] = new
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK]   {rel}")


def patch_block(rel: str, old: str, new: str) -> None:
    """整块替换。new 已存在则跳过；old 出现次数 != 1 则报错。"""
    p = ROOT / rel
    s = p.read_text(encoding="utf-8")
    if new in s:
        print(f"  [跳过] {rel} : 已是目标状态")
        return
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"  [失败] {rel}: 锚点出现 {n} 次（应为 1）\n---\n{old}\n---")
    p.write_text(s.replace(old, new, 1), encoding="utf-8")
    print(f"  [OK]   {rel}")


# =====================================================================
# 1. models/dsanet.py —— 末尾追加 MinIPOnly 类
# =====================================================================
MINIP_CLASS = '''

class MinIPOnly(nn.Module):
    """论文 Table VI 的 MinIP-only 基线：砍掉 TEB / TF / STF，只留 SEB + Decoder。

    用途：判断「与论文的差距」出在基础配方还是时序模块。
    forward 签名与 DSANet 一致，所以 Dataset / 训练循环 / 评估脚本都不用改。
    """

    def __init__(self,
                 in_ch: int = 1,
                 chs: tuple[int, ...] = CHS,
                 n_classes: int = 3,
                 deep_supervision: bool = True,
                 pool_first_layer: bool = False,
                 grad_checkpoint: bool = False) -> None:
        super().__init__()
        self.seb = SpatialEncodingBranch(in_ch, chs, pool_first_layer, grad_checkpoint)
        self.decoder = Decoder(
            fused_ch=chs[-1],          # 512，不是 1024：没有 STF 做 concat
            enc_chs=chs, n_classes=n_classes,
            deep_supervision=deep_supervision, grad_checkpoint=grad_checkpoint,
        )

    def forward(self, seq: torch.Tensor, minip: torch.Tensor
                ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # seq 收到但不用 —— 保持签名一致，训练 / 评估代码零改动
        feat_m, skips_m = self.seb(minip, injections=None)
        return self.decoder(feat_m, skips_m)
'''

_p = ROOT / "models/dsanet.py"
_s = _p.read_text(encoding="utf-8")
if "class MinIPOnly" in _s:
    print("  [跳过] models/dsanet.py : 已有 MinIPOnly")
else:
    _p.write_text(_s.rstrip("\n") + "\n" + MINIP_CLASS, encoding="utf-8")
    print("  [OK]   models/dsanet.py : 追加 MinIPOnly")


# =====================================================================
# 2. models/__init__.py
# =====================================================================
patch_line("models/__init__.py", "from .dsanet import DSANet",
           lambda ln: ln if "MinIPOnly" in ln
           else ln.replace("import DSANet", "import DSANet, MinIPOnly", 1))

patch_line("models/__init__.py", '"DSANet",',
           lambda ln: ln if "MinIPOnly" in ln
           else ln.replace('"DSANet",', '"DSANet", "MinIPOnly",', 1))


# =====================================================================
# 3. train.py
# =====================================================================
patch_line("train.py", "from models.dsanet import DSANet",
           lambda ln: ln if "MinIPOnly" in ln
           else ln.replace("import DSANet", "import DSANet, MinIPOnly", 1))

patch_block("train.py",
            '''def build_model(args) -> DSANet:
    return DSANet(
        num_frames=args.num_frames,
        tf_layers=args.tf_layers,
        tf_heads=args.tf_heads,
        alpha_mode=args.alpha_mode,
        attn_backend=args.attn_backend,
        grad_checkpoint=args.grad_checkpoint,
        deep_supervision=not args.no_deep_supervision,
    )''',
            '''def build_model(args):
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
    )''')

_ARCH_ARG = (
    '    ap.add_argument("--arch", default="dsanet", choices=["dsanet", "minip_only"],\n'
    '                    help="网络结构。dsanet = 论文主模型（双输入 + 时序模块）；"\n'
    '                         "minip_only = 论文 Table VI 的 MinIP-only 基线"\n'
    '                         "（只有 SEB + Decoder，用于判断差距出在基础还是时序模块）")\n'
)
patch_line("train.py", 'ap.add_argument("--fold", type=int, default=0)',
           lambda ln: ln if "--arch" in ln else ln + "\n" + _ARCH_ARG.rstrip("\n"))

patch_line("train.py", 'log(f"  alpha_mode      : {args.alpha_mode}")',
           lambda ln: ln if "架构" in ln
           else ln + '\n    log(f"  架构            : {args.arch}")')


# =====================================================================
# 4. evaluate.py
# =====================================================================
patch_line("evaluate.py", "from models.dsanet import DSANet",
           lambda ln: ln if "MinIPOnly" in ln
           else ln.replace("import DSANet", "import DSANet, MinIPOnly", 1))

patch_block("evaluate.py",
            '''def build_from_ckpt(ckpt: dict, device) -> tuple[DSANet, dict]:
    targs = ckpt.get("args", {}) or {}
    model = DSANet(
        num_frames=targs.get("num_frames", 8),
        tf_layers=targs.get("tf_layers", 4),
        tf_heads=targs.get("tf_heads", 8),
        alpha_mode=targs.get("alpha_mode", "attention"),
        attn_backend=targs.get("attn_backend", "sdpa"),
        grad_checkpoint=False,                 # 推理不需要
        deep_supervision=not targs.get("no_deep_supervision", False),

    )''',
            '''def build_from_ckpt(ckpt: dict, device):
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
        )''')

print("\n全部完成。")
