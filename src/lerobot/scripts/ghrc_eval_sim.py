#!/usr/bin/env python3
"""GHRC 2026 评估系统 — 仿真评测容器入口。

负责 Isaac Sim 仿真、机器人控制、observation 采集、episode 生命周期管理与断言评测。
通过 WebSocket 连接推理容器获取 action，不直接加载 policy。

使用示例:

```bash
python -m lerobot.scripts.ghrc_eval_sim
python -m lerobot.scripts.ghrc_eval_sim --config eval_config/eval_sim.yaml --task task4
```
"""

from __future__ import annotations

import argparse
import logging
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.lerobot.utils.control_utils import init_keyboard_listener
from src.lerobot.sim_eval import (
    EpisodeResult,
    InferenceSummary,
    NoOpAssertion,
    TASK_CHOICES,
    TASK_DEFAULT_MAX_STEPS,
    TASK_DEFAULT_CONFIG_PATH,
    TASK_DEFAULT_TEXT,
    action_to_dict,
    build_assertion_args_from_task_yaml,
    build_robot,
    create_task_assertion,
    flatten_obs,
    load_container_config,
    load_task_yaml_config,
    print_summary,
    require_config_keys,
    resolve_device,
    save_episode,
    save_summary,
)
from src.lerobot.sim_eval.protocol import (
    DEFAULT_CONNECTION_TIMEOUT,
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_WEBSOCKET_MAX_SIZE_BYTES,
    DEFAULT_WEBSOCKET_CONTROL_PORT,
    DEFAULT_WEBSOCKET_STREAM_PORT,
)
from src.lerobot.sim_eval.websocket_client import EvalWebSocketClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SimEvalContainer:
    """`sim-eval` 容器运行时主类。

    类属性:
        args: 配置对象。
        task: 当前任务名。
        task_text: 当前任务描述。
        max_steps: 单个 episode 最大步数。
        num_episodes: 总评测 episode 数。
        log_dir: 结果输出目录。
        robot: 仿真机器人对象。
        ws_client: infer 容器客户端。
        assertion_fn: 任务断言对象。
        summary: 批量评测汇总对象。
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """初始化 `sim-eval` 运行时。

        Args:
            args: 已解析配置。
        """

        self.args = args
        self.task = str(args.task).lower()
        self.task_text = str(args.task_text)
        self.max_steps = int(args.max_steps)
        self.num_episodes = int(args.num_episodes)
        self.log_dir = Path(args.log_dir)
        self.device = resolve_device(str(args.device))

        self.robot: Any | None = None
        self.ws_client: EvalWebSocketClient | None = None
        self.assertion_fn: Any | None = None
        self.summary: InferenceSummary | None = None
        self.task_yaml_config: dict[str, Any] | None = None

    def initialize(self) -> None:
        """初始化机器人、评测断言和通信客户端。"""

        logger.info("任务：%s | 设备：%s", self.task, self.device)
        logger.info(
            "infer 连接：%s:%s/%s",
            self.args.sim_infer_host,
            self.args.sim_infer_control_port,
            self.args.sim_infer_stream_port,
        )

        self.robot = build_robot(self.task, str(self.args.task_config_path))
        self.ws_client = EvalWebSocketClient(
            sim_infer_host=str(self.args.sim_infer_host),
            control_port=int(self.args.sim_infer_control_port),
            stream_port=int(self.args.sim_infer_stream_port),
            heartbeat_interval=float(self.args.heartbeat_interval),
            connection_timeout=float(self.args.connection_timeout),
            websocket_max_size_bytes=getattr(
                self.args,
                "websocket_max_size_bytes",
                DEFAULT_WEBSOCKET_MAX_SIZE_BYTES,
            ),
            transfer_log_every_steps=int(getattr(self.args, "print_every", 20)),
            color_logs=bool(getattr(self.args, "color_logs", True)),
        )
        self.task_yaml_config = load_task_yaml_config(str(self.args.task_config_path))

        self.assertion_fn = (
            create_task_assertion(
                self.task,
                build_assertion_args_from_task_yaml(
                    self.task,
                    self.task_yaml_config,
                    base_args=self.args,
                ),
            )
            if bool(getattr(self.args, "enable_assertion", False))
            else NoOpAssertion()
        )
        self.summary = InferenceSummary(
            task_name=self.task,
            policy_type="remote",
            policy_path=str(getattr(self.args, "policy_path", "remote://infer")),
            num_episodes=self.num_episodes,
            max_steps=self.max_steps,
            task_text=self.task_text,
        )

    def run(self) -> None:
        """运行完整评测流程。"""

        if self.robot is None or self.ws_client is None or self.summary is None:
            raise RuntimeError("SimEvalContainer 尚未初始化")

        logger.info("连接机器人...")
        self.robot.connect()
        for _ in range(5):
            self.robot.step(render=True)

        logger.info("连接 infer 容器...")
        self.ws_client.connect()

        listener, events = init_keyboard_listener()
        try:
            self._wait_for_user_start(events)
            episode_idx = 0
            while episode_idx < self.num_episodes:
                should_advance = self._run_episode(episode_idx, events)
                if should_advance:
                    episode_idx += 1
        finally:
            save_summary(self.log_dir, self.summary)
            print_summary(self.summary)
            try:
                self.ws_client.send_stop()
            except Exception:
                pass
            self.cleanup()
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    pass

    def _wait_for_user_start(self, events: dict[str, Any]) -> None:
        """等待用户按下 Enter 开始评测。

        Args:
            events: 键盘监听事件字典。
        """

        if self.robot is None:
            raise RuntimeError("robot 尚未初始化")

        if getattr(self.args, 'auto_start', False):
            logger.info("auto_start 模式，跳过等待 Enter")
            return

        logged = False
        events["start_record"] = False
        while not events.get("start_record", False):
            self.robot.step(render=True)
            if not logged:
                logger.info("按 Enter 键开始...")
                logged = True

        logger.info("开始评估，共 %s 个 episodes", self.num_episodes)

    def _run_episode(self, episode_idx: int, events: dict[str, Any]) -> bool:
        """运行单个 episode。

        Args:
            episode_idx: 当前 episode 编号。
            events: 键盘监听事件字典。

        Returns:
            bool: True 表示当前 episode 计入统计并推进到下一个 episode；
            False 表示当前 episode 需要重录，不推进计数。
        """

        if self.robot is None or self.ws_client is None or self.assertion_fn is None or self.summary is None:
            raise RuntimeError("运行时组件未初始化")

        result = EpisodeResult(episode_id=episode_idx)
        step = 0
        logger.info("开始 Episode %s/%s", episode_idx + 1, self.num_episodes)

        try:
            self.ws_client.send_start(episode_id=episode_idx)

            while step < self.max_steps:
                observation = self.robot.get_observation()
                # 将扁平观测转为 LeRobot 标准格式后发送给 infer
                observation = flatten_obs(observation)
                self.ws_client.send_observation(
                    observation=observation,
                    task=self.task,
                    episode_id=episode_idx,
                    step=step,
                )

                action_message = self.ws_client.wait_for_action(
                    episode_id=episode_idx,
                    step=step,
                    timeout=float(getattr(self.args, "action_wait_timeout", 10.0)),
                )
                if action_message is None:
                    raise TimeoutError(f"等待 infer action 超时：episode={episode_idx} step={step}")

                action = self._decode_action(action_message.get("action"))
                action_dict = action_to_dict(action)
                self.robot.send_action(action_dict)

                step += 1
                result.steps = step

                is_success, terminal, reason, metrics = self.assertion_fn(
                    self.robot,
                    step,
                    action=action,
                    extra_info={"max_steps": self.max_steps},
                )
                result.assertion_result = reason
                result.metrics = metrics
                result.score = self._extract_score(metrics)

                if step % int(getattr(self.args, "print_every", 20)) == 0:
                    act_mae = action.abs().mean().item()
                    logger.info(
                        "[Episode %s/%s] step=%s/%s | Act_MAE: %.4f | Score: %s | %s",
                        episode_idx + 1,
                        self.num_episodes,
                        step,
                        self.max_steps,
                        act_mae,
                        result.score,
                        reason,
                    )

                if events.get("rerecord_episode", False):
                    events["rerecord_episode"] = False
                    logger.info("[Episode %s] 重新录制当前 episode", episode_idx + 1)
                    self._reset_robot()
                    self.ws_client.send_episode_end(
                        episode_id=episode_idx,
                        status="rerecord",
                        reason="用户请求重录当前 episode",
                        metrics=result.metrics,
                        step=step,
                    )
                    return False

                if is_success:
                    result.status = "success"
                    result.finalize()
                    logger.info("[Episode %s] SUCCESS: %s", episode_idx + 1, reason)
                    self.summary.add_result(result)
                    save_episode(self.log_dir, result)
                    self._reset_robot()
                    self.ws_client.send_episode_end(
                        episode_id=episode_idx,
                        status="success",
                        reason=reason,
                        metrics=result.metrics,
                        step=step,
                    )
                    return True

                if terminal:
                    result.status = "failed"
                    result.metrics["failure_reason"] = reason
                    result.metrics["episode_end_status"] = self._derive_terminal_status(reason)
                    result.finalize()
                    logger.info("[Episode %s] FAILED: %s", episode_idx + 1, reason)
                    self.summary.add_result(result)
                    save_episode(self.log_dir, result)
                    self._reset_robot()
                    self.ws_client.send_episode_end(
                        episode_id=episode_idx,
                        status=result.metrics["episode_end_status"],
                        reason=reason,
                        metrics=result.metrics,
                        step=step,
                    )
                    return True

            result.status = "failed"
            result.metrics["failure_reason"] = f"达到最大步数 {self.max_steps}"
            result.metrics["episode_end_status"] = "timeout"
            result.finalize()
            logger.info("[Episode %s] TIMEOUT: 达到最大步数", episode_idx + 1)
            self.summary.add_result(result)
            save_episode(self.log_dir, result)
            self._reset_robot()
            self.ws_client.send_episode_end(
                episode_id=episode_idx,
                status="timeout",
                reason=result.metrics["failure_reason"],
                metrics=result.metrics,
                step=step,
            )
            return True
        except Exception as exc:
            logger.error("[Episode %s] 异常: %s", episode_idx + 1, exc)
            logger.error(traceback.format_exc())
            result.status = "error"
            result.error_msg = str(exc)
            result.metrics["episode_end_status"] = "error"
            result.finalize()
            self.summary.add_result(result)
            save_episode(self.log_dir, result)
            try:
                self._reset_robot()
            except Exception:
                pass
            try:
                self.ws_client.send_episode_end(
                    episode_id=episode_idx,
                    status="error",
                    reason=str(exc),
                    metrics=result.metrics,
                    step=step,
                )
            except Exception:
                pass
            return True

    def _reset_robot(self) -> None:
        """执行 episode 间的本地仿真 reset。"""

        if self.robot is None:
            return
        for _ in range(int(getattr(self.args, "reset_retries", 3))):
            self.robot.reset()

    def _decode_action(self, action: Any) -> torch.Tensor:
        """将消息中的 action 转换为可发送给 robot 的张量。

        Args:
            action: 消息中的 action 负载。

        Returns:
            torch.Tensor: 归一化后的动作张量。
        """

        if isinstance(action, torch.Tensor):
            tensor_action = action.detach().float().cpu()
        elif isinstance(action, np.ndarray):
            tensor_action = torch.from_numpy(action).float()
        elif isinstance(action, list):
            tensor_action = torch.tensor(action, dtype=torch.float32)
        else:
            raise ValueError(f"不支持的 action 类型：{type(action)}")

        tensor_action = tensor_action.reshape(-1)
        if self.task == "task4" and tensor_action.numel() == 18:
            padding = torch.zeros(2, dtype=tensor_action.dtype)
            tensor_action = torch.cat([tensor_action, padding], dim=0)
        return tensor_action

    def _extract_score(self, metrics: dict[str, Any]) -> int:
        """从任务指标中提取统一 score 字段。

        Args:
            metrics: 任务断言返回的指标字典。

        Returns:
            int: 当前 episode score。
        """

        if self.task == "task1":
            return int(metrics.get("task1_total_score", 0))
        if self.task == "task2":
            return int(metrics.get("task2_total_score", 0))
        if self.task == "task3":
            return int(metrics.get("task3_total_score", 0))
        if self.task == "task4":
            return int(metrics.get("task4_total_score", 0))
        return int(metrics.get("total_score", 0))

    def _derive_terminal_status(self, reason: str) -> str:
        """根据失败原因归一化 infer 侧接收的终止状态。

        Args:
            reason: 断言返回的失败原因。

        Returns:
            str: `failed` 或 `timeout`。
        """

        timeout_tokens = ("超时", "达到最大步数")
        return "timeout" if any(token in reason for token in timeout_tokens) else "failed"

    def cleanup(self) -> None:
        """释放 WebSocket 与 robot 资源。"""

        if self.ws_client is not None:
            try:
                self.ws_client.disconnect()
            except Exception:
                pass
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    """解析命令行参数并加载 YAML 配置。

    Returns:
        argparse.Namespace: 配置对象。
    """

    parser = argparse.ArgumentParser(description="sim-eval 容器入口")
    parser.add_argument("--config", type=str, default="eval_config/eval_sim.yaml", help="配置文件路径")
    parser.add_argument("--task", type=str, default=None, choices=TASK_CHOICES, help="覆盖任务名")
    cli_args = parser.parse_args()

    args = load_container_config(
        config_path=cli_args.config,
        defaults={
            "device": "auto",
            "num_episodes": 100,
            "print_every": 20,
            "log_dir": "logs/sim_eval_container",
            "enable_assertion": False,
            "action_wait_timeout": 10.0,
            "reset_retries": 3,
            "sim_infer_host": "sim-infer",
            "sim_infer_control_port": DEFAULT_WEBSOCKET_CONTROL_PORT,
            "sim_infer_stream_port": DEFAULT_WEBSOCKET_STREAM_PORT,
            "heartbeat_interval": DEFAULT_HEARTBEAT_INTERVAL,
            "connection_timeout": DEFAULT_CONNECTION_TIMEOUT,
            "websocket_max_size_bytes": None,
            "task_config_path": None,
        },
        path_fields={"task_config_path", "log_dir", "policy_path"},
    )

    require_config_keys(args, ["task", "num_episodes", "log_dir"], "sim-eval 容器配置")

    # --task CLI 参数覆盖 YAML 中的 task
    if cli_args.task:
        args.task = str(cli_args.task)

    args.task = str(args.task).lower()
    if not getattr(args, "task_config_path", None):
        args.task_config_path = str(Path(TASK_DEFAULT_CONFIG_PATH[args.task]).resolve())
    args.task_text = str(getattr(args, "task_text", "") or TASK_DEFAULT_TEXT.get(args.task, args.task))
    args.max_steps = int(getattr(args, "max_steps", TASK_DEFAULT_MAX_STEPS.get(args.task, 1000)))
    return args


def main() -> None:
    """程序主入口。"""

    args = parse_args()
    logger.info("=" * 60)
    logger.info("Sim-Eval 容器启动")
    logger.info("=" * 60)
    logger.info("配置文件：%s", args.config)
    logger.info("任务：%s", args.task)
    logger.info("任务描述：%s", args.task_text)
    logger.info("Episodes: %s | 最大步数：%s", args.num_episodes, args.max_steps)
    logger.info("=" * 60)

    runtime = SimEvalContainer(args)
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
