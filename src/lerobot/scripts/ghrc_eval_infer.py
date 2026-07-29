#!/usr/bin/env python3
"""GHRC 2026 评估系统 — 推理容器入口。

负责接收仿真容器发送的 observation、执行模型推理、返回 action，
episode 结束时调用 policy reset。

使用示例:

```bash
python -m lerobot.scripts.ghrc_eval_infer
python -m lerobot.scripts.ghrc_eval_infer --config eval_config/eval_infer.yaml --task task4
```
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from src.lerobot.sim_eval import (
    TASK_CHOICES,
    TASK_DEFAULT_CONFIG_PATH,
    TASK_DEFAULT_MAX_STEPS,
    TASK_DEFAULT_POLICY_PATH,
    TASK_DEFAULT_TEXT,
    InferenceContext,
    ResetContext,
    create_policy_adapter,
    load_container_config,
    require_config_keys,
    resolve_device,
)
from src.lerobot.sim_eval.policy_adapter import PolicyAdapter
from src.lerobot.sim_eval.protocol import (
    DEFAULT_CONNECTION_TIMEOUT,
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_WEBSOCKET_CONTROL_PORT,
    DEFAULT_WEBSOCKET_MAX_SIZE_BYTES,
    DEFAULT_WEBSOCKET_SERVER_HOST,
    DEFAULT_WEBSOCKET_STREAM_PORT,
    get_observation_from_message,
)
from src.lerobot.sim_eval.websocket_server import SimInferWebSocketServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SimInferContainer:
    """`infer` 容器运行时主类。

    类属性:
        args: 配置对象。
        task: 当前任务名。
        task_text: 当前任务描述。
        device: 运行设备。
        adapter: 策略适配器。
        ws_server: WebSocket 服务端。
        observation_queue: 待处理 observation 队列。
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """初始化 `infer` 容器运行时对象。

        Args:
            args: 已解析配置。
        """

        self.args = args
        self.task = str(args.task).lower()
        self.task_text = str(args.task_text)
        self.device = resolve_device(str(args.device))

        self.running = False
        self.stop_requested = False
        self.episode_active = False
        self.current_episode_id = 0

        self.adapter: PolicyAdapter | None = None
        self.ws_server: SimInferWebSocketServer | None = None
        self.observation_queue: queue.Queue[dict[str, Any]] = queue.Queue()

        self._pending_reset: ResetContext | None = None
        self._reset_lock = threading.Lock()

    def initialize(self) -> None:
        """初始化 adapter 与 WebSocket 服务。

        Returns:
            无。
        """

        logger.info("任务：%s | 设备：%s", self.task, self.device)
        logger.info("初始化策略适配器...")

        adapter_type = str(getattr(self.args, "adapter_type", "lerobot"))
        adapter_class = getattr(self.args, "adapter_class", None)
        adapter_config = getattr(self.args, "adapter_config", {}) or {}

        self.adapter = create_policy_adapter(adapter_type=adapter_type, adapter_class=adapter_class)
        adapter_runtime_config = {
            **adapter_config,
            "policy_type": getattr(self.args, "policy_type", None),
            "task": self.task,
            "task_text": self.task_text,
        }
        self.adapter.load(
            model_path=str(getattr(self.args, "policy_path", "")),
            device=self.device,
            config=adapter_runtime_config,
        )

        logger.info("初始化 WebSocket 服务...")
        self.ws_server = SimInferWebSocketServer(
            host=str(getattr(self.args, "websocket_server_host", DEFAULT_WEBSOCKET_SERVER_HOST)),
            control_port=int(getattr(self.args, "websocket_control_port", DEFAULT_WEBSOCKET_CONTROL_PORT)),
            stream_port=int(getattr(self.args, "websocket_stream_port", DEFAULT_WEBSOCKET_STREAM_PORT)),
            heartbeat_interval=float(getattr(self.args, "heartbeat_interval", DEFAULT_HEARTBEAT_INTERVAL)),
            connection_timeout=float(getattr(self.args, "connection_timeout", DEFAULT_CONNECTION_TIMEOUT)),
            websocket_max_size_bytes=getattr(
                self.args,
                "websocket_max_size_bytes",
                DEFAULT_WEBSOCKET_MAX_SIZE_BYTES,
            ),
        )
        self.ws_server.set_start_callback(self._on_start)
        self.ws_server.set_reset_callback(self._on_reset)
        self.ws_server.set_episode_end_callback(self._on_episode_end)
        self.ws_server.set_stop_callback(self._on_stop)
        self.ws_server.set_observation_callback(self._on_observation)

    def _on_start(self, data: dict[str, Any]) -> None:
        """处理 start 控制命令。

        Args:
            data: start 控制消息。
        """

        self.current_episode_id = int(data.get("episode_id", 0))
        self.episode_active = True
        logger.info("收到 start 命令：episode=%s", self.current_episode_id)

    def _on_reset(self, data: dict[str, Any]) -> None:
        """处理兼容保留的 reset 控制命令。

        Args:
            data: reset 控制消息。
        """

        reset_context = ResetContext(
            episode_id=int(data.get("episode_id", self.current_episode_id)),
            status="reset",
            reason="收到 reset 控制命令",
            metrics=data.get("data", {}) or {},
            step=int(data.get("step", 0)),
        )
        self._set_pending_reset(reset_context)
        self.episode_active = False
        self._drain_observation_queue()
        logger.info("收到 reset 命令：episode=%s", reset_context.episode_id)

    def _on_episode_end(self, data: dict[str, Any]) -> None:
        """处理统一的 episode_end 控制命令。

        Args:
            data: episode_end 控制消息。
        """

        reset_context = ResetContext(
            episode_id=int(data.get("episode_id", self.current_episode_id)),
            status=str(data.get("status", "reset")),
            reason=str(data.get("reason", "")),
            metrics=data.get("metrics", {}) or {},
            step=int(data.get("step", 0)),
        )
        if bool(data.get("need_reset", True)):
            self._set_pending_reset(reset_context)
        self.episode_active = False
        self._drain_observation_queue()
        logger.info(
            "收到 episode_end 命令：episode=%s status=%s",
            reset_context.episode_id,
            reset_context.status,
        )

    def _on_stop(self) -> None:
        """处理 stop 控制命令。"""

        logger.info("收到 stop 命令")
        self.stop_requested = True
        self.running = False

    def _on_observation(self, data: dict[str, Any]) -> None:
        """将 observation 消息加入待处理队列。

        Args:
            data: stream observation 消息。
        """

        self.observation_queue.put(data)

    def _drain_observation_queue(self) -> None:
        """清空 observation 队列，丢弃边界期间残留帧。"""

        dropped = 0
        while True:
            try:
                self.observation_queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        if dropped > 0:
            logger.info("已丢弃残留 observation：%s", dropped)

    def _set_pending_reset(self, reset_context: ResetContext) -> None:
        """登记待执行 reset 请求。

        Args:
            reset_context: reset 上下文。
        """

        with self._reset_lock:
            self._pending_reset = reset_context

    def _pop_pending_reset(self) -> ResetContext | None:
        """弹出待执行的 reset 请求。

        Returns:
            ResetContext | None: 待执行 reset，上下文不存在时返回 None。
        """

        with self._reset_lock:
            reset_context = self._pending_reset
            self._pending_reset = None
        return reset_context

    def run(self) -> None:
        """启动服务并进入推理主循环。"""

        if self.adapter is None or self.ws_server is None:
            raise RuntimeError("SimInferContainer 尚未初始化")

        self.running = True
        self.ws_server.start()
        logger.info("等待 sim-eval 侧 start 命令...")

        try:
            with torch.inference_mode():
                while self.running and not self.stop_requested:
                    self._apply_pending_reset_if_needed()

                    try:
                        observation_message = self.observation_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    # stream/control 为双通道，取到 observation 后再检查一次 reset，
                    # 避免新 episode 首帧在 reset 前被推理。
                    self._apply_pending_reset_if_needed()

                    if not self.episode_active:
                        logger.debug("当前 episode 未激活，丢弃 observation")
                        continue

                    observation_episode = int(
                        observation_message.get("episode_id", self.current_episode_id)
                    )
                    if observation_episode != self.current_episode_id:
                        logger.debug(
                            "丢弃过期 observation：msg_episode=%s current_episode=%s",
                            observation_episode,
                            self.current_episode_id,
                        )
                        continue

                    self._process_observation(observation_message)
        finally:
            self.cleanup()

    def _apply_pending_reset_if_needed(self) -> None:
        """执行挂起的 policy reset。"""

        if self.adapter is None:
            return
        reset_context = self._pop_pending_reset()
        if reset_context is None:
            return
        self.adapter.reset(reset_context)
        logger.info(
            "已执行 policy reset：episode=%s status=%s",
            reset_context.episode_id,
            reset_context.status,
        )

    def _process_observation(self, observation_message: dict[str, Any]) -> None:
        """处理单条 observation 并回传 action。

        Args:
            observation_message: stream observation 消息。
        """

        if self.adapter is None or self.ws_server is None:
            raise RuntimeError("运行时组件未初始化")

        observation = get_observation_from_message(observation_message)
        context = InferenceContext(
            task=self.task,
            task_text=self.task_text,
            episode_id=int(observation_message.get("episode_id", self.current_episode_id)),
            step=int(observation_message.get("step", 0)),
            timestamp=float(observation_message.get("timestamp", time.time())),
        )
        action = self.adapter.predict(observation=observation, context=context)
        self.ws_server.broadcast_action(
            action=action,
            episode_id=context.episode_id,
            step=context.step,
            task=self.task,
        )

    def cleanup(self) -> None:
        """停止服务并释放 adapter 资源。"""

        self.running = False

        if self.ws_server is not None:
            self.ws_server.stop()

        if self.adapter is not None:
            self.adapter.close()


def parse_args() -> argparse.Namespace:
    """解析命令行参数并加载 YAML 配置。

    Returns:
        argparse.Namespace: 已完成默认值填充与路径解析的配置对象。
    """

    parser = argparse.ArgumentParser(description="infer 容器入口")
    parser.add_argument("--config", type=str, default="eval_config/eval_infer.yaml", help="配置文件路径")
    parser.add_argument("--task", type=str, default=None, choices=TASK_CHOICES, help="覆盖任务名")
    cli_args = parser.parse_args()

    args = load_container_config(
        config_path=cli_args.config,
        defaults={
            "device": "auto",
            "adapter_type": "lerobot",
            "adapter_class": None,
            "adapter_config": {},
            "task_policy_paths": {},
            "require_task_policy_paths": False,
            "websocket_server_host": DEFAULT_WEBSOCKET_SERVER_HOST,
            "websocket_control_port": DEFAULT_WEBSOCKET_CONTROL_PORT,
            "websocket_stream_port": DEFAULT_WEBSOCKET_STREAM_PORT,
            "heartbeat_interval": DEFAULT_HEARTBEAT_INTERVAL,
            "connection_timeout": DEFAULT_CONNECTION_TIMEOUT,
            "websocket_max_size_bytes": None,
            "task_config_path": None,
        },
        path_fields={"policy_path", "task_config_path"},
    )

    require_config_keys(args, ["task"], "infer 容器配置")

    # --task CLI 参数覆盖 YAML 中的 task
    if cli_args.task:
        args.task = str(cli_args.task)

    args.task = str(args.task).lower()
    if not getattr(args, "task_config_path", None):
        args.task_config_path = str(Path(TASK_DEFAULT_CONFIG_PATH[args.task]).resolve())
    args.task_text = str(getattr(args, "task_text", "") or TASK_DEFAULT_TEXT.get(args.task, args.task))
    args.max_steps = int(getattr(args, "max_steps", TASK_DEFAULT_MAX_STEPS.get(args.task, 1000)))
    adapter_type = str(getattr(args, "adapter_type", "lerobot")).lower()
    if adapter_type == "lerobot" and not getattr(args, "adapter_class", None):
        require_config_keys(args, ["policy_type"], "infer 容器配置")
        _resolve_policy_path_for_task(args)
        if not getattr(args, "policy_path", None) and args.task in TASK_DEFAULT_POLICY_PATH:
            args.policy_path = str(Path(TASK_DEFAULT_POLICY_PATH[args.task]))
        require_config_keys(args, ["policy_type", "policy_path"], "infer 容器配置")
    return args


def _resolve_policy_path_for_task(args: argparse.Namespace) -> None:
    """Resolve task-specific policy path overrides from YAML.

    `run_eval.sh all` reuses one infer YAML and only changes `--task`.
    A single `policy_path` cannot point to four task checkpoints, so YAML may
    provide `task_policy_paths: {task1: ..., task2: ...}`. Values are resolved
    relative to the config file directory, matching `load_container_config`
    behavior for top-level path fields.
    """

    task_policy_paths = getattr(args, "task_policy_paths", {}) or {}
    if not isinstance(task_policy_paths, dict):
        raise ValueError("infer 容器配置 task_policy_paths 必须是字典")

    config_dir = Path(str(args.config)).expanduser().resolve().parent
    task = str(args.task).lower()
    selected_path = task_policy_paths.get(task) or getattr(args, "policy_path", None)
    if not selected_path:
        if bool(getattr(args, "require_task_policy_paths", False)):
            raise ValueError(
                f"infer 容器配置要求显式 policy 路径，但 task_policy_paths 未提供当前任务：{task}"
            )
        return

    selected = Path(str(selected_path)).expanduser()
    args.policy_path = str(selected if selected.is_absolute() else (config_dir / selected).resolve())


def main() -> None:
    """程序主入口。"""

    args = parse_args()
    logger.info("=" * 60)
    logger.info("Infer 容器启动")
    logger.info("=" * 60)
    logger.info("配置文件：%s", args.config)
    logger.info("任务：%s", args.task)
    logger.info("任务描述：%s", args.task_text)
    logger.info("WebSocket 端口：control=%s stream=%s", args.websocket_control_port, args.websocket_stream_port)
    logger.info("=" * 60)

    runtime = SimInferContainer(args)
    try:
        runtime.initialize()
        runtime.run()
    except KeyboardInterrupt:
        logger.info("收到中断信号，退出...")
    except Exception as exc:
        logger.error("运行异常：%s", exc)
        logger.error(traceback.format_exc())
        runtime.cleanup()


if __name__ == "__main__":
    main()
