#!/usr/bin/env bash
set -Eeuo pipefail

# ======== 按需修改 ========
USER="wangwx"                     # 远程统一用户名
PASS="NT0c8ipNMx"                 # 固定密码
PORT=22
HOSTS=(
  "192.168.4.23" "192.168.4.35" "192.168.4.41" "192.168.4.42"
  "192.168.4.43" "192.168.4.44" "192.168.4.55" "192.168.4.56"
  "192.168.4.57" "192.168.4.58" "192.168.4.61" "192.168.4.62"
  "192.168.4.63" "192.168.4.64" "192.168.4.65" "192.168.4.66"
  "192.168.4.67" "192.168.4.68"
)
# ===========================

SSH_OPTS=(-p "$PORT" -o ConnectTimeout=6 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
SCP_OPTS=(-P "$PORT" -o ConnectTimeout=6 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

WORK="$(mktemp -d /tmp/sshmesh.XXXXXX)"
ALL_KEYS="$WORK/authorized_keys.all"; : >"$ALL_KEYS"
ALL_KNOWN="$WORK/known_hosts.all";   : >"$ALL_KNOWN"

ok_collect=()  fail_collect=()  nohome=()
ok_push=()     fail_push=()     skip_push=()

can_ssh() { sshpass -p "$PASS" ssh "${SSH_OPTS[@]}" "$USER@$1" 'echo ok' >/dev/null 2>&1; }

ensure_ssh_dir() {
  # 创建 ~/.ssh + 生成 ed25519（如缺失）
  sshpass -p "$PASS" ssh "${SSH_OPTS[@]}" "$USER@$1" '
    set -e
    # 若家目录不存在或无权限，下面会失败，返回非零
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"
    if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
      ssh-keygen -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519" >/dev/null
    fi
  ' >/dev/null 2>&1
}

echo "[1/3] 确保各节点有 ~/.ssh & 公钥，收集公钥与 hostkey ..."
for H in "${HOSTS[@]}"; do
  echo "  -> $H"
  if ! can_ssh "$H"; then
    echo "     [SKIP] 无法连接或认证失败"; fail_collect+=("$H"); continue
  fi
  if ! ensure_ssh_dir "$H"; then
    echo "     [SKIP] 无法创建 ~/.ssh（该用户家目录可能不存在/不可写）"; nohome+=("$H"); continue
  fi

  # 收集公钥与已有 authorized_keys
  sshpass -p "$PASS" ssh "${SSH_OPTS[@]}" "$USER@$H" \
    'cat ~/.ssh/id_ed25519.pub 2>/dev/null;
     cat ~/.ssh/id_rsa.pub      2>/dev/null;
     cat ~/.ssh/authorized_keys 2>/dev/null;' >> "$WORK/all_keys.tmp" || true

  # 收集 host key（用于后面分发 known_hosts，减少交互提示）
  ssh-keyscan -T 5 -H "$H" >> "$WORK/known_hosts.tmp" 2>/dev/null || true

  ok_collect+=("$H")
done

if [[ -s "$WORK/all_keys.tmp" ]]; then
  awk 'NF' "$WORK/all_keys.tmp" | sort -u > "$ALL_KEYS"
else
  echo "没有收集到任何公钥，退出。"; exit 1
fi
[[ -s "$WORK/known_hosts.tmp" ]] && sort -u "$WORK/known_hosts.tmp" > "$ALL_KNOWN"

echo "[2/3] 分发合并后的 authorized_keys ..."
for H in "${HOSTS[@]}"; do
  echo "  -> $H"
  if ! can_ssh "$H"; then
    echo "     [SKIP] 无法连接"; skip_push+=("$H"); continue
  fi

  TMP="/tmp/authorized_keys.$USER.$$"
  if ! sshpass -p "$PASS" scp "${SCP_OPTS[@]}" "$ALL_KEYS" "$USER@$H:$TMP" >/dev/null 2>&1; then
    echo "     [FAIL] scp 失败"; fail_push+=("$H"); continue
  fi

  if sshpass -p "$PASS" ssh "${SSH_OPTS[@]}" "$USER@$H" \
      "mkdir -p ~/.ssh && chmod 700 ~/.ssh &&
       [ -f ~/.ssh/authorized_keys ] && cp ~/.ssh/authorized_keys ~/.ssh/authorized_keys.bak || true &&
       mv '$TMP' ~/.ssh/authorized_keys &&
       chmod 600 ~/.ssh/authorized_keys" >/dev/null 2>&1; then
    ok_push+=("$H")
  else
    echo "     [FAIL] 覆盖 authorized_keys 失败（家目录/权限问题？）"
    fail_push+=("$H")
  fi
done

echo "[3/3] 分发 known_hosts（可选，减少首次连接提示） ..."
if [[ -s "$ALL_KNOWN" ]]; then
  for H in "${HOSTS[@]}"; do
    sshpass -p "$PASS" scp "${SCP_OPTS[@]}" "$ALL_KNOWN" "$USER@$H:/tmp/known_hosts.$$" >/dev/null 2>&1 || continue
    sshpass -p "$PASS" ssh "${SSH_OPTS[@]}" "$USER@$H" \
      'mkdir -p ~/.ssh && chmod 700 ~/.ssh &&
       mv /tmp/known_hosts.* ~/.ssh/known_hosts 2>/dev/null || true &&
       chmod 644 ~/.ssh/known_hosts' >/dev/null 2>&1 || true
  done
fi

echo "==== 汇总 ===="
echo "准备成功: ${#ok_collect[@]} 台: ${ok_collect[*]:-}"
echo "准备失败(连不上): ${#fail_collect[@]} 台: ${fail_collect[*]:-}"
echo "家目录/权限问题: ${#nohome[@]} 台: ${nohome[*]:-}"
echo "分发成功: ${#ok_push[@]} 台: ${ok_push[*]:-}"
echo "分发失败: ${#fail_push[@]} 台: ${fail_push[*]:-}"
echo "分发跳过(连不上): ${#skip_push[@]} 台: ${skip_push[*]:-}"
echo "合并后的 authorized_keys 存在: $ALL_KEYS"
echo "提示：若某些主机家的目录本身不存在，需要管理员先创建家目录，再重跑脚本。"
