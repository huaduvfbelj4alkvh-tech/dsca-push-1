#!/bin/bash
# ---------------------------------------------------------------------------
# 训练进度看板（在集群上跑，单文件、无依赖）
#
# 用法:
#   bash watch_train.sh                          # 自动扫 runs/logs/*.log 全部
#   bash watch_train.sh f0_axfix_768             # 只看一个（多打 val / 日志尾）
#   bash watch_train.sh f0_axfix_768 f1_axfix_768 f2_axfix_768 f3_axfix_768 f4_axfix_768
#
# 刷新间隔用环境变量 EVERY（秒，默认 1200 = 20 分钟）:
#   EVERY=600 bash watch_train.sh
#
# 项目根用 ROOT（默认 $HOME/dsca）。
# 退出: Ctrl+C（全部收到收尾横幅时会自动退出）
#
# 设计要点:
#   * 进度用「最后一行的 step 行」反推（epoch / step / 累计秒 → s/iter），
#     所以能反映"卡被抢导致即时变慢"，而不是拿开跑时的平均速度硬外推。
#   * 同时打印「已完成 epoch 行数」，两者应互相印证。
#   * 不用 step s/100 当进度 —— 那个 /100 是固定分母，与数据集大小无关。
# ---------------------------------------------------------------------------

ROOT="${ROOT:-$HOME/dsca}"
EVERY="${EVERY:-1200}"

cd "$ROOT" || { echo "找不到项目根: $ROOT"; exit 1; }

if [ "$#" -gt 0 ]; then
    RUNS="$*"
else
    for f in runs/logs/*.log; do
        [ -f "$f" ] && RUNS="$RUNS $(basename "$f" .log)"
    done
fi
RUNS=$(echo $RUNS)

if [ -z "$RUNS" ]; then
    echo "runs/logs 下没有任何 .log"; exit 1
fi
NRUN=$(echo $RUNS | wc -w)

echo "项目根 : $ROOT"
echo "监控   : ${NRUN} 个 run（每 ${EVERY}s 刷一次，Ctrl+C 退出）"
echo

while true; do
    echo "===== $(date '+%m-%d %H:%M') ====="
    NFIN=0
    for RUN in $RUNS; do
        LOG="runs/logs/$RUN.log"

        if [ ! -f "$LOG" ]; then
            echo "  $RUN   ⚠️ 还没有日志"
            continue
        fi

        if pgrep -f -- "--out runs/$RUN" >/dev/null; then
            ALIVE=存活
        else
            ALIVE=停笔
        fi

        TOT=$(grep -m1 'epoch × iter' "$LOG" | grep -oE '[0-9]+' | head -1)
        IPE=$(grep -m1 'epoch × iter' "$LOG" | grep -oE '[0-9]+' | tail -1)
        IPE="${IPE:-100}"
        DONE=$(grep -c '>>> epoch' "$LOG")
        LAST=$(grep -E 'step [0-9]+/[0-9]+' "$LOG" | tail -1)

        if [ -n "$LAST" ] && [ -n "$TOT" ]; then
            e=$(printf '%s' "$LAST" | sed -E 's/.*epoch ([0-9]+)\/[0-9]+.*/\1/')
            s=$(printf '%s' "$LAST" | sed -E 's/.*step ([0-9]+)\/[0-9]+.*/\1/')
            c=$(printf '%s' "$LAST" | sed -E 's/.*\(([0-9]+)s\).*/\1/')
            awk -v n="$RUN" -v a="$ALIVE" -v D="$DONE" -v e="$e" -v s="$s" -v c="$c" \
                -v I="$IPE" -v E="$TOT" 'BEGIN{
                spi = (s > 0 ? c / s : 0);
                ds = (e - 1) * I + s; tt = E * I;
                tag = (a == "存活" ? "★存活" : "  停笔");
                if (spi > 0) {
                    printf "  %-16s %s  %3d/%-3d %5.1f%%  %5.1fs/ep  剩余 %5.2fh  完成 %s\n",
                        n, tag, D, E, 100 * ds / tt, spi * I,
                        (tt - ds) * spi / 3600, strftime("%H:%M", systime() + (tt - ds) * spi);
                } else {
                    printf "  %-16s %s  %3d/%-3d   （暂无 step 行）\n", n, tag, D, E;
                }
            }'
        else
            echo "  $RUN   $ALIVE   还没有 step 行"
        fi

        if [ "$NRUN" -eq 1 ]; then
            V=$(grep 'val Dice' "$LOG" | tail -1)
            echo "      最近 val : ${V:-（还没有；第一次 val 在 epoch 10）}"
            echo "      日志尾   : $(tail -1 "$LOG")"
        fi

        grep -q '训练结束' "$LOG" && NFIN=$((NFIN + 1))
    done

    if [ "$NFIN" -eq "$NRUN" ]; then
        echo
        echo "✅ ${NFIN}/${NRUN} 个 run 都已收工（日志里有收尾横幅），看板退出。"
        exit 0
    fi

    echo
    sleep "$EVERY"
done
