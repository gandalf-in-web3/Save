#!/usr/bin/env bash
# 用法示例：
#   bash ddp.sh --n_process 16 --machines 192.168.4.23 192.168.4.35 192.168.4.55 \
#       --mixed_precision bf16 train.py --lr 1e-4 --batch_size 8
#
# 说明：
# - 第一个机器为主节点（rank 0），显示它的进度条。
# - ssh 已免密，统一用户名为 wangwx（如需改，改下面 REMOTE_USER）。
# - Ctrl-C 会在所有机器上杀掉相关进程（train.py/accelerate/torchrun）。

set -euo pipefail

REMOTE_USER="wangwx"
PORT="${PORT:-29500}"
RUN_ID="ddp_$$_$RANDOM"

usage() {
  echo "Usage: bash ddp.sh --n_process N --machines ip1 ip2 ... [accelerate_args] train.py [train_args]"
  exit 1
}

# -------- 解析参数（保持简单）--------
NPROC=""
machines=()
accel_args=()
train_args=()
script=""

if [[ $# -lt 1 ]]; then usage; fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n_process)
      NPROC="$2"; shift 2 ;;
    --machines)
      shift
      # 收集 IP，直到遇到下一个 --参数 或 *.py
      while [[ $# -gt 0 && "$1" != --* && $1 != *.py ]]; do
        machines+=("$1"); shift
      done
      ;;
    *.py)
      script="$1"; shift
      train_args=("$script" "$@")
      break
      ;;
    *)
      accel_args+=("$1"); shift ;;
  esac
done

[[ -z "${NPROC:-}" || -z "${script:-}" || ${#machines[@]} -eq 0 ]] && usage

NMACH=${#machines[@]}

# -------- Ctrl-C 清理 --------
cleanup() {
  echo -e "\n[ddp.sh] Stopping all remote processes..."
  for ((i=0;i<NMACH;i++)); do
    host="${machines[$i]}"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${REMOTE_USER}@${host}" "bash -lc '
      pidfile=/tmp/${RUN_ID}_${i}.pid
      if [[ -f \$pidfile ]]; then
        pgid=\$(cat \$pidfile)
        kill -TERM -\$pgid 2>/dev/null || true
        sleep 1
        kill -KILL -\$pgid 2>/dev/null || true
      fi
    '" || true
  done
}
trap 'cleanup; exit 130' INT TERM

# -------- 启动 worker（后台无输出）--------
for ((i=1;i<NMACH;i++)); do
  host="${machines[$i]}"
  cmd=(NCCL_SOCKET_IFNAME=ib0 accelerate launch
       --num_processes "$NPROC"
       --num_machines "$NMACH"
       --machine_rank "$i"
       --main_process_ip "${machines[0]}"
       --main_process_port "$PORT")
  cmd+=("${accel_args[@]}")
  cmd+=("${train_args[@]}")

  # 组装远程命令并后台运行
  remote_cmd=$(printf "%q " "${cmd[@]}")
  echo "[ddp.sh] start worker rank=$i on $host with nproc=$NPROC"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${REMOTE_USER}@${host}" \
    "log=/tmp/${RUN_ID}_${i}.log; pid=/tmp/${RUN_ID}_${i}.pid; : >\"\$log\"; \
    nohup setsid bash -lc \"$remote_cmd\" >>\"\$log\" 2>&1 & echo \$! >\"\$pid\"" &
done

# 同时在本机跟随所有 worker 的日志（后台），带前缀区分
for ((i=1;i<NMACH;i++)); do
  host="${machines[$i]}"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${REMOTE_USER}@${host}" \
    "tail -n +1 -F /tmp/${RUN_ID}_${i}.log" | sed -u "s/^/[rank ${i}@${host}] /" &
done

# -------- 启动 master（前台显示进度条）--------
master="${machines[0]}"
cmd0=(NCCL_SOCKET_IFNAME=ib0 accelerate launch
      --num_processes "$NPROC"
      --num_machines "$NMACH"
      --machine_rank 0
      --main_process_ip "${machines[0]}"
      --main_process_port "$PORT")
cmd0+=("${accel_args[@]}")
cmd0+=("${train_args[@]}")
remote_cmd0=$(printf "%q " "${cmd0[@]}")

echo "[ddp.sh] start master rank=0 on $master with nproc=$NPROC"
# 前台 attach 到主机输出
ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${REMOTE_USER}@${master}" \
  "log=/tmp/${RUN_ID}_0.log; pid=/tmp/${RUN_ID}_0.pid; : >\"\$log\"; \
   nohup setsid bash -lc \"$remote_cmd0\" >>\"\$log\" 2>&1 & echo \$! >\"\$pid\""

ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${REMOTE_USER}@${master}" \
  "tail -n +1 -f /tmp/${RUN_ID}_0.log"

# 正常结束就退出；如 Ctrl-C，会走 trap 清理
echo "[ddp.sh] done."

sleep 60
