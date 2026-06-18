#!/usr/bin/env bash
# GHRC 评测编排器 — crontab 入口脚本
#
# 用法:
#   将此脚本配置到 crontab 中实现每日定时评测。
#   使用 flock 防止多个编排实例并发执行。
#
# crontab 示例 (每天凌晨 2:00):
#   0 2 * * * /path/to/project/scripts/crontab_entry.sh >> /var/log/eval_orchestrator.log 2>&1
#
# 环境变量要求（应在 crontab 或外部文件中设置）:
#   FEISHU_APP_ID        飞书应用 App ID
#   FEISHU_APP_SECRET    飞书应用 App Secret
#   DOCKER_USERNAME      Docker registry 用户名
#   DOCKER_PASSWORD      Docker registry 密码

set -euo pipefail

# ---- 项目根目录 ----
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- 锁文件 ----
LOCKFILE="${LOCKFILE:-/tmp/eval_orchestrator.lock}"
exec 200>"${LOCKFILE}"
if ! flock -n 200; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 已有评测编排进程运行中，退出"
    exit 0
fi

# ---- 凭据校验 ----
if [[ -z "${FEISHU_APP_ID:-}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: 未设置 FEISHU_APP_ID 环境变量"
    exit 1
fi
if [[ -z "${FEISHU_APP_SECRET:-}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: 未设置 FEISHU_APP_SECRET 环境变量"
    exit 1
fi
if [[ -z "${DOCKER_USERNAME:-}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: 未设置 DOCKER_USERNAME 环境变量"
    exit 1
fi
if [[ -z "${DOCKER_PASSWORD:-}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: 未设置 DOCKER_PASSWORD 环境变量"
    exit 1
fi

# ---- 执行 ----
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始评测编排..."
cd "${PROJECT_DIR}"

python -m lerobot.scripts.ghrc_eval_orchestrator \
    --config "${EVAL_ORCHESTRATOR_CONFIG:-eval_config/eval_orchestrator.yaml}" \
    --project-dir "${PROJECT_DIR}"

EXIT_CODE=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 评测编排完成，退出码=${EXIT_CODE}"
exit ${EXIT_CODE}
