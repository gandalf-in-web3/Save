#!/usr/bin/env bash

# 用于在多个机器上启动glances的脚本

PASS="NT0c8ipNMx"   # 统一 SSH 密码
USER="wangwx"       # 登录用户
PORT=22             # SSH 端口
GLANCES_PORT=61209  # glances -s 默认端口
SESSION=${SESSION:-glances-nav}

HOSTS=(
  "192.168.4.23" "192.168.4.35" "192.168.4.41" "192.168.4.42"
  "192.168.4.43" "192.168.4.44" "192.168.4.48" "192.168.4.55"
  "192.168.4.56" "192.168.4.57" "192.168.4.58" "192.168.4.61"
  "192.168.4.62" "192.168.4.63" "192.168.4.64" "192.168.4.65"
  "192.168.4.66" "192.168.4.67" "192.168.4.68"
)

# 启动glances服务端
for h in "${HOSTS[@]}"; do
  printf "%-15s : " "$h"
  sshpass -p "$PASS" ssh -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -T "${USER}@${h}" \
    "nohup glances -s -B 0.0.0.0 -p $GLANCES_PORT >/dev/null 2>&1 &" \
    && echo started || echo unreachable
done

# 启动glances监视
usage() {
  cat <<'USAGE'
Usage:
  ./glances-switcher.sh             # 使用脚本内的 HOSTS 数组
  GLANCES_PORT=61209 ./glances-switcher.sh
Env:
  SESSION         tmux 会话名（默认 glances-nav）
  GLANCES_PORT    glances -s 的默认端口（默认 61209）
Keys (全局，无需前缀):
  Up / Down       切换上一/下一台机器
  Home / End      跳到第一/最后一台
  Ctrl+R          重新拉起当前窗格里的 glances
  Ctrl+Q          退出并销毁整个会话
备注：↑/↓ 被用来切换窗口，glances 内部的箭头滚动将不可用
USAGE
}

if [[ "${1:-}" =~ ^-h|--help$ ]]; then usage; exit 0; fi
command -v tmux >/dev/null || { echo "[ERR] tmux 未安装"; exit 1; }
command -v glances >/dev/null || { echo "[ERR] glances 未安装"; exit 1; }

# 允许从文件读取 HOSTS（可选）：每行一个 host 或 host:port；忽略空行与以#开头的行
if [[ "${1:-}" == "-f" && -n "${2:-}" ]]; then
  mapfile -t HOSTS < <(grep -vE '^\s*(#|$)' "$2" | xargs -I{} echo {})
fi

if [ ${#HOSTS[@]} -eq 0 ]; then
  echo "[ERR] HOSTS 为空。请在脚本顶部填入 HOSTS=(...) 或用 -f 指定文件。"
  exit 1
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true

# 如果会话已存在则直接附着
if tmux has-session -t "$SESSION" 2>/dev/null; then
  exec tmux attach -t "$SESSION"
fi

# 生成启动命令（支持 host:port）
cmd_for() {
  local hp="$1" host port
  if [[ "$hp" == *:* ]]; then host="${hp%%:*}"; port="${hp##*:}"; else host="$hp"; port="$GLANCES_PORT"; fi
  printf 'printf "Connecting to %s:%s...\n"; exec glances -c %q -p %q --disable-plugin alert' \
    "$host" "$port" "$host" "$port"
}

# 创建会话 + 第一个窗口
first="${HOSTS[0]}"
tmux new-session -d -s "$SESSION" -n "$first" "$(cmd_for "$first")"

# 其余窗口
for hp in "${HOSTS[@]:1}"; do
  tmux new-window -t "$SESSION" -n "$hp" "$(cmd_for "$hp")"
done

# 体验优化：窗口标题=host；退出时保留窗口内容；状态栏更清晰
tmux set-window-option -t "$SESSION" remain-on-exit on
tmux set-option  -t "$SESSION" mouse on
tmux set-option  -t "$SESSION" status on
tmux set-option  -t "$SESSION" status-right '#[fg=colour247]#{session_name}  [#{window_index}+1/#{session_windows}]  %H:%M '

# 绑定无前缀按键：↑/↓ 切换窗口，Home/End 跳首尾，Ctrl+R 重启当前 pane，Ctrl+Q 退出
tmux bind-key -n Up            previous-window
tmux bind-key -n Down          next-window
tmux bind-key -n Home          select-window -t :0
tmux bind-key -n End           last-window
tmux bind-key -n C-r           respawn-pane -k
tmux bind-key -n C-q           kill-session

tmux display-message -t "$SESSION" "Hints: ↑/↓ 切换主机；Ctrl+R 重启当前；Ctrl+Q 退出会话。"

exec tmux attach -t "$SESSION"
