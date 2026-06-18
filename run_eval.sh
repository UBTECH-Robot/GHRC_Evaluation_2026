#!/usr/bin/env bash
# Usage: ./run_eval.sh task4 | all
# Dual-container evaluation:
#   - infer (CPU) exposes WebSocket services
#   - sim-eval (GPU) connects to infer and runs Isaac Sim evaluation

set -Eeuo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="${1:-all}"
ALL_TASKS=("task4" "task1" "task2" "task3")
[[ "${TASK}" == "all" ]] && TASKS=("${ALL_TASKS[@]}") || TASKS=("${TASK}")

INFER_IMAGE="${INFER_IMAGE:-ghrc-eval-infer:latest}"
SIM_IMAGE="${SIM_IMAGE:-ghrc-eval-sim:latest}"
HOST_WS="${SCRIPT_DIR}"
CONTAINER_WS="/workspace/eval"
INFER_CONFIG="${INFER_CONFIG:-eval_config/eval_infer.yaml}"
SIM_CONFIG="${SIM_CONFIG:-eval_config/eval_sim.yaml}"

ISAAC_CACHE="${ISAAC_CACHE_ROOT:-${HOME}/.cache/isaac_sim_container}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
HEADLESS="${HEADLESS:-1}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
INFER_READY_TIMEOUT="${INFER_READY_TIMEOUT:-300}"
RUNTIME_BOOTSTRAP="${RUNTIME_BOOTSTRAP:-1}"

DEFAULT_LOG_ROOT="${HOME}/.cache/challenge_baseline_runner"
LOG_ROOT="${LOG_ROOT:-${DEFAULT_LOG_ROOT}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_LOG_DIR="${LOG_ROOT}/${RUN_ID}"

LOG_INCLUDE_PATTERN='Sim-Eval 容器启动|配置文件：|任务：|Episodes:|连接机器人|连接 infer|auto_start|按 Enter|开始评估|开始 Episode|Episode.*step=|SUCCESS|FAILED|TIMEOUT|成功|失败|异常|汇总|Score|Connected|INIT|等待 infer action 超时|运行异常'
LOG_EXCLUDE_PATTERN='Warning.*usd|omni\.physicsschema'

PYTHON="/isaac-sim/python.sh"
INFER_PY="source /isaac-sim/setup_python_env.sh && LD_PRELOAD=/isaac-sim/kit/libcarb.so /isaac-sim/kit/python/bin/python3"

XHOST_OPENED=0
ACTIVE_CONTAINERS=()

read_yaml_scalar() {
    local file="$1"
    local key="$2"
    local default_value="$3"
    local value

    value="$(
        awk -F':' -v key="${key}" '
            $1 ~ "^[[:space:]]*" key "[[:space:]]*$" {
                sub(/^[[:space:]]+/, "", $2)
                sub(/[[:space:]]+$/, "", $2)
                print $2
                exit
            }
        ' "${file}" 2>/dev/null || true
    )"

    [[ -n "${value}" ]] && echo "${value}" || echo "${default_value}"
}

is_port_listening() {
    local port="$1"

    if command -v ss &>/dev/null; then
        ss -lnt "( sport = :${port} )" 2>/dev/null | tail -n +2 | grep -q ":${port}[[:space:]]"
    else
        netstat -lnt 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"
    fi
}

add_container() {
    local name="$1"
    ACTIVE_CONTAINERS+=("${name}")
}

cleanup() {
    local exit_code=$?
    local name

    for (( i=${#ACTIVE_CONTAINERS[@]}-1; i>=0; i-- )); do
        name="${ACTIVE_CONTAINERS[$i]}"
        docker rm -f "${name}" &>/dev/null || true
    done

    if [[ "${HEADLESS}" == "0" && "${XHOST_OPENED}" == "1" ]]; then
        xhost -local:docker &>/dev/null 2>&1 || true
        XHOST_OPENED=0
    fi

    if (( exit_code != 0 )); then
        error "评测失败，运行日志目录: ${RUN_LOG_DIR}"
    fi
}

trap cleanup EXIT INT TERM

print_log_tail() {
    local label="$1"
    local log_file="$2"

    if [[ -f "${log_file}" ]]; then
        error "${label} 日志尾部 (${log_file}):"
        tail -n 80 "${log_file}" >&2 || true
    else
        error "未找到 ${label} 日志: ${log_file}"
    fi
}

wait_for_infer_ready() {
    local task="$1"
    local infer_name="$2"
    local control_port="$3"
    local stream_port="$4"
    local timeout_seconds="$5"
    local launcher_pid="$6"
    local infer_log_file="$7"
    local start_ts

    start_ts="$(date +%s)"

    while true; do
        if docker inspect -f '{{.State.Running}}' "${infer_name}" >/dev/null 2>&1; then
            :
        elif kill -0 "${launcher_pid}" 2>/dev/null; then
            if (( $(date +%s) - start_ts >= timeout_seconds )); then
                error "等待 infer 容器 ${infer_name} 创建超时 (${timeout_seconds}s)"
                print_log_tail "infer" "${infer_log_file}"
                return 1
            fi
            sleep 1
            continue
        else
            error "infer 容器 ${infer_name} 已退出"
            print_log_tail "infer" "${infer_log_file}"
            return 1
        fi

        if is_port_listening "${control_port}" && is_port_listening "${stream_port}"; then
            info "infer 已就绪: control=${control_port}, stream=${stream_port}"
            return 0
        fi

        if (( $(date +%s) - start_ts >= timeout_seconds )); then
            error "等待 infer 就绪超时 (${timeout_seconds}s)，端口 ${control_port}/${stream_port} 未监听"
            print_log_tail "infer" "${infer_log_file}"
            return 1
        fi

        sleep 2
    done
}

runtime_bootstrap_cmd() {
    if [[ "${RUNTIME_BOOTSTRAP}" == "1" ]]; then
        printf '%s' "${PYTHON} -m pip install -i ${PIP_MIRROR} -e . --no-deps -q; "
        printf '%s' "${PYTHON} -m pip install -i ${PIP_MIRROR} lz4 msgpack websockets pillow -q; "
    fi
}

run_sim_with_logging() {
    local sim_name="$1"
    local sim_args="$2"
    local sim_log_file="$3"
    local sim_status
    local pipe_status

    set +e
    set +o pipefail

    docker run --rm --name "${sim_name}" \
        --entrypoint /bin/bash \
        --privileged --network host --user root --gpus all --shm-size=8g \
        -v "${HOST_WS}:${CONTAINER_WS}:rw" -w "${CONTAINER_WS}" \
        -v "${ISAAC_CACHE}/cache/kit:/root/.cache/kit:rw" \
        -v "${ISAAC_CACHE}/cache/ov:/root/.cache/ov:rw" \
        -v "${ISAAC_CACHE}/cache/pip:/root/.cache/pip:rw" \
        -v "${ISAAC_CACHE}/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
        -v "${ISAAC_CACHE}/cache/computecache:/root/.cache/nvidia/ComputeCache:rw" \
        -v "${HF_CACHE}:/root/.cache/huggingface:rw" \
        -e NO_AT_BRIDGE=1 \
        -e ACCEPT_EULA=Y \
        -e PRIVACY_CONSENT=Y \
        -e XDG_RUNTIME_DIR=/tmp \
        -e "PYTHONPATH=${CONTAINER_WS}" \
        -e DISPLAY="${DISPLAY:-}" \
        -e QT_X11_NO_MITSHM=1 \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        "${SIM_IMAGE}" \
        -c "cd ${CONTAINER_WS}; $(runtime_bootstrap_cmd) ${PYTHON} -m lerobot.scripts.ghrc_eval_sim ${sim_args}" 2>&1 \
        | tee "${sim_log_file}" \
        | grep --line-buffered -E "${LOG_INCLUDE_PATTERN}" \
        | grep --line-buffered -vE "${LOG_EXCLUDE_PATTERN}"

    pipe_status=("${PIPESTATUS[@]}")
    sim_status="${pipe_status[0]}"

    set -o pipefail
    set -e

    if [[ "${sim_status}" -ne 0 ]]; then
        error "sim-eval 容器运行失败，退出码: ${sim_status}"
        print_log_tail "sim-eval" "${sim_log_file}"
        return "${sim_status}"
    fi

    return 0
}

prepare_environment() {
    command -v docker &>/dev/null || { error "未找到 docker"; exit 1; }
    docker image inspect "${INFER_IMAGE}" &>/dev/null 2>&1 || { error "镜像 ${INFER_IMAGE} 不存在"; exit 1; }
    docker image inspect "${SIM_IMAGE}"   &>/dev/null 2>&1 || { error "镜像 ${SIM_IMAGE} 不存在"; exit 1; }

    for d in cache/kit cache/ov cache/pip cache/glcache cache/computecache data documents; do
        mkdir -p "${ISAAC_CACHE}/${d}"
    done
    mkdir -p "${HF_CACHE}"

    if ! mkdir -p "${RUN_LOG_DIR}" 2>/dev/null; then
        error "日志目录不可写: ${RUN_LOG_DIR}"
        error "请设置可写的 LOG_ROOT，例如: export LOG_ROOT=\$HOME/.cache/challenge_baseline_runner"
        exit 1
    fi

    if [[ "${HEADLESS}" == "0" ]]; then
        xhost +local:docker &>/dev/null 2>&1 || true
        XHOST_OPENED=1
    fi
}

run_eval() {
    local task="$1"
    local infer_name="eval_infer_${task}"
    local sim_name="eval_sim_${task}"
    local infer_args="--config ${INFER_CONFIG} --task ${task}"
    local sim_args="--config ${SIM_CONFIG} --task ${task}"
    local task_log_dir="${RUN_LOG_DIR}/${task}"
    local infer_log_file="${task_log_dir}/infer.log"
    local sim_log_file="${task_log_dir}/sim.log"
    local infer_launcher_pid
    local infer_control_port
    local infer_stream_port

    infer_control_port="$(read_yaml_scalar "${INFER_CONFIG}" "websocket_control_port" "8765")"
    infer_stream_port="$(read_yaml_scalar "${INFER_CONFIG}" "websocket_stream_port" "8766")"

    mkdir -p "${task_log_dir}"
    docker rm -f "${infer_name}" "${sim_name}" &>/dev/null || true
    add_container "${infer_name}"
    add_container "${sim_name}"

    info "==========================================="
    info "评估: ${task}"
    info "运行目录: ${task_log_dir}"
    info "==========================================="

    info "启动 infer 容器..."
    docker run --rm --name "${infer_name}" \
        --entrypoint /bin/bash \
        --privileged --network host --user root --shm-size=8g \
        -v "${HOST_WS}:${CONTAINER_WS}:rw" -w "${CONTAINER_WS}" \
        "${INFER_IMAGE}" \
        -c "cd ${CONTAINER_WS}; $(runtime_bootstrap_cmd) ${INFER_PY} -m lerobot.scripts.ghrc_eval_infer ${infer_args}" \
        &> "${infer_log_file}" &
    infer_launcher_pid=$!

    info "等待 infer 就绪..."
    wait_for_infer_ready \
        "${task}" \
        "${infer_name}" \
        "${infer_control_port}" \
        "${infer_stream_port}" \
        "${INFER_READY_TIMEOUT}" \
        "${infer_launcher_pid}" \
        "${infer_log_file}"

    info "infer 日志: ${infer_log_file}"
    info "sim-eval 日志: ${sim_log_file}"
    info "启动 sim-eval 容器..."
    run_sim_with_logging "${sim_name}" "${sim_args}" "${sim_log_file}"

    docker rm -f "${infer_name}" &>/dev/null || true
    info "${task} 完成"
}

main() {
    prepare_environment

    info "infer: ${INFER_IMAGE}  |  sim: ${SIM_IMAGE}  |  任务: ${TASKS[*]}"
    info "run_id: ${RUN_ID}"
    info "日志根目录: ${RUN_LOG_DIR}"

    if [[ "${RUNTIME_BOOTSTRAP}" == "1" ]]; then
        warn "RUNTIME_BOOTSTRAP=1: 运行时会执行 pip install。生产环境建议将依赖预装进镜像后设为 0。"
    fi

    for task in "${TASKS[@]}"; do
        run_eval "${task}"
    done

    info "全部完成"
    info "运行日志目录: ${RUN_LOG_DIR}"
}

main "$@"
