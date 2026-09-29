#!/usr/bin/env bash
# 把集群上各实验目录里的 history.json（每个几十 KB）拉到本机，
# 之后就能在**本机**出图、调窗口、反复改样式，不用来回 scp 图片。
#
# 用法（Git Bash，在 F:/Projects/DSCA 下）：：
#     bash fetch_histories.sh
#     bash fetch_histories.sh --also-eval        # 连 eval_val/eval_test.json 一起拉
#     RUNS="lc_n36 5fold_adam_f0" bash fetch_histories.sh
#     PORT=<port> USER_R=<user> HOST=<cluster-host> bash fetch_histories.sh
#
# 只拉 json/小文件，**不拉** .pth（几十 MB 级，本机出图用不上）。
set -u

# 🔴 必填：本脚本不内置任何集群地址/账号/端口，请自己传进来。
#    例：PORT=<port> USER_R=<user> HOST=<cluster-host> bash fetch_histories.sh
PORT="${PORT:?请设置 PORT（SSH 端口）}"
USER_R="${USER_R:?请设置 USER_R（集群登录用户名）}"
HOST="${HOST:?请设置 HOST（集群登录地址）}"
REMOTE="${USER_R}@${HOST}"
RUNS="${RUNS:-lc_n36 lc_n72 lc_n108 5fold_adam_f0 5fold_adam_f1 5fold_adam_f2 5fold_adam_f3r 5fold_adam_f4}"
ALSO_EVAL=0
for a in "$@"; do
  [ "$a" = "--also-eval" ] && ALSO_EVAL=1
done

# 脚本所在目录（不用 dirname：某些 Git Bash 的 PATH 里没有它）
SELF_DIR="$(cd "${0%/*}" 2>/dev/null && pwd)"
[ -z "${SELF_DIR:-}" ] && SELF_DIR="$(pwd)"
cd "$SELF_DIR" || exit 1
echo "本机项目目录: $SELF_DIR"

# ---- 0) 连通性 + 免密检查（不通就给出唯一正确的修法） ----
if ! ssh -o BatchMode=yes -o ConnectTimeout=12 -p "$PORT" "$REMOTE" "echo ok" >/dev/null 2>&1; then
  echo
  echo "❌ 免密登录未生效，拉不了文件。常见三种原因："
  echo "   1) 公钥还没装到服务器（最常见）→ Git Bash 跑这一条（需输一次服务器密码）："
  echo "      ssh -p $PORT $REMOTE \"mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys\" < ~/.ssh/id_ed25519.pub"
  echo "   2) 本机压根没有私钥 → 先 ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N \"\""
  echo "   3) 没连校园网/VPN → 到 $HOST 的路由不通"
  echo
  echo "   临时替代（不想装密钥）：用 MobaXterm / FileZilla（SFTP，端口 $PORT）"
  echo "   把 ~/dsca/runs/<名字>/history.json 手动拖到本机 runs/<名字>/ 下即可。"
  exit 1
fi
echo "✅ 免密连通 $REMOTE"

# ---- 1) 逐个拉 history.json ----
mkdir -p runs
ok=0; miss=0; skip=0
for r in $RUNS; do
  d="runs/$r"
  mkdir -p "$d"
  if scp -q -P "$PORT" "$REMOTE:~/dsca/runs/$r/history.json" "$d/history.json" 2>/dev/null \
     && [ -s "$d/history.json" ]; then
    sz=$(wc -c < "$d/history.json" | tr -d ' ')
    echo "  ✓ $r/history.json    ${sz} bytes"
    ok=$((ok + 1))
    if [ "$ALSO_EVAL" = "1" ]; then
      if scp -q -P "$PORT" "$REMOTE:~/dsca/runs/$r/eval_*.json" "$d/" 2>/dev/null; then
        echo "      + eval_*.json"
      fi
    fi
  else
    # 拉失败时不要留空目录，否则出图会把"已存在的空 run"和"根本没这个 run"混淆
    [ -f "$d/history.json" ] || rmdir "$d" 2>/dev/null
    echo "  ✗ $r   （集群上没这个目录，或还没产出 history.json）"
    miss=$((miss + 1))
  fi
done

echo
echo "完成：成功 $ok / 失败 $miss"
echo
echo "现在在本机直接出图（零参数，默认 spec 就指向刚拉回来的这几个）："
echo "    \"E:/Anaconda/envs/dsca/python.exe\" plot_learning_curve.py"
echo "    \"E:/Anaconda/envs/dsca/python.exe\" plot_learning_curve.py --smooth 10 --tail 100"
echo
echo "判读前先看脚本打印的 sigma：曲线差距 < sigma 的一律按「不可分辨」处理。"
