"""WebSocket Client。

本文件用于 `sim-eval` 容器连接 `infer` 容器，负责发送控制命令、推送 observation，
以及接收 infer 返回的 action 数据。

使用示例:

```python
from lerobot.sim_eval.websocket_client import EvalWebSocketClient

client = EvalWebSocketClient(sim_infer_host="sim-infer")
client.connect()
client.send_start(episode_id=0)
client.send_observation(
    observation={"observation.state": [0.0] * 20},
    task="task4",
    episode_id=0,
    step=0,
)
action_message = client.wait_for_action(episode_id=0, step=0, timeout=5.0)
client.send_stop()
client.disconnect()
```
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import threading
import time
from typing import Any, Callable

try:
    _ws_sync_client = importlib.import_module("websockets.sync.client")
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "未找到 websockets.sync.client，请安装 websockets>=11：pip install -U websockets"
    ) from exc

ClientConnection = _ws_sync_client.ClientConnection
connect = _ws_sync_client.connect
_ws_exceptions = importlib.import_module("websockets.exceptions")
ConnectionClosed = _ws_exceptions.ConnectionClosed
ConnectionClosedOK = getattr(_ws_exceptions, "ConnectionClosedOK", ConnectionClosed)

from .protocol import (
    DEFAULT_CONNECTION_TIMEOUT,
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_WEBSOCKET_MAX_SIZE_BYTES,
    DEFAULT_WEBSOCKET_CONTROL_PORT,
    DEFAULT_WEBSOCKET_STREAM_PORT,
    build_control_message,
    build_episode_end_message,
    build_stream_message,
    decode_stream_message,
    encode_stream_message,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("websockets.client").setLevel(logging.WARNING)


class EvalWebSocketClient:
    """`sim-eval` 容器使用的同步 WebSocket 客户端。

    类属性:
        sim_infer_host: infer 容器主机名。
        control_port: 控制命令端口。
        stream_port: 双向数据流端口。
        control_uri: 控制通道 URI。
        stream_uri: 数据流通道 URI。
    """

    def __init__(
        self,
        sim_infer_host: str = "sim-infer",
        control_port: int = DEFAULT_WEBSOCKET_CONTROL_PORT,
        stream_port: int = DEFAULT_WEBSOCKET_STREAM_PORT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT,
        websocket_max_size_bytes: int | None = DEFAULT_WEBSOCKET_MAX_SIZE_BYTES,
        transfer_log_every_steps: int = 20,
        color_logs: bool = True,
    ) -> None:
        """初始化客户端。

        Args:
            sim_infer_host: infer 容器主机名。
            control_port: Control 端口。
            stream_port: Stream 端口。
        """

        self.sim_infer_host = sim_infer_host
        self.control_port = int(control_port)
        self.stream_port = int(stream_port)
        self.heartbeat_interval = float(heartbeat_interval)
        self.connection_timeout = float(connection_timeout)
        self.websocket_max_size_bytes = (
            None if websocket_max_size_bytes is None else int(websocket_max_size_bytes)
        )
        self.transfer_log_every_steps = max(1, int(transfer_log_every_steps))
        self.color_logs = bool(color_logs)
        self._supports_color = self.color_logs and os.getenv("TERM") not in {None, "", "dumb"}
        self.control_uri = f"ws://{sim_infer_host}:{control_port}"
        self.stream_uri = f"ws://{sim_infer_host}:{stream_port}"

        self._control_connection: ClientConnection | None = None
        self._stream_connection: ClientConnection | None = None
        self._connected = False
        self._running = False
        self._stream_thread: threading.Thread | None = None

        self._on_observation_callback: Callable[[dict[str, Any]], None] | None = None
        self._on_action_callback: Callable[[dict[str, Any]], None] | None = None

        self._latest_observation: dict[str, Any] | None = None
        self._latest_action: dict[str, Any] | None = None
        self._data_lock = threading.Lock()
        self._data_condition = threading.Condition(self._data_lock)
        self._control_lock = threading.Lock()
        self._stream_send_lock = threading.Lock()

        self._freq_lock = threading.Lock()
        self._freq_window_start = time.monotonic()
        self._tx_observation_count = 0
        self._tx_observation_bytes = 0
        self._rx_action_count = 0
        self._rx_action_bytes = 0

    def set_observation_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """设置 observation 接收回调。

        Args:
            callback: 接收 stream observation 消息的回调函数。
        """

        self._on_observation_callback = callback

    def set_action_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """设置 action 接收回调。

        Args:
            callback: 接收 stream action 消息的回调函数。
        """

        self._on_action_callback = callback

    def connect(self) -> None:
        """连接 infer 容器的 control 与 stream 通道。"""

        try:
            logger.info("连接到 Control 端口：%s", self.control_uri)
            self._control_connection = connect(
                self.control_uri,
                open_timeout=self.connection_timeout,
                close_timeout=self.connection_timeout,
                max_size=self.websocket_max_size_bytes,
            )
            logger.info("Control 连接成功")

            logger.info("连接到 Stream 端口：%s", self.stream_uri)
            self._stream_connection = connect(
                self.stream_uri,
                open_timeout=self.connection_timeout,
                close_timeout=self.connection_timeout,
                max_size=self.websocket_max_size_bytes,
            )
            logger.info("Stream 连接成功")

            self._connected = True
            self._running = True
            self._stream_thread = threading.Thread(
                target=self._receive_stream_messages,
                daemon=True,
            )
            self._stream_thread.start()
        except Exception as exc:
            logger.error("连接失败：%s", exc)
            self._connected = False
            raise

    def disconnect(self) -> None:
        """断开 control 与 stream 通道。"""

        self._running = False
        self._connected = False

        if self._control_connection is not None:
            try:
                self._control_connection.close()
            except Exception:
                pass

        if self._stream_connection is not None:
            try:
                self._stream_connection.close()
            except Exception:
                pass

        logger.info("已断开连接")

    def _receive_stream_messages(self) -> None:
        """接收 infer 侧发来的 stream 消息。"""

        while self._running and self._stream_connection is not None:
            try:
                message = self._stream_connection.recv()
                if message is None:
                    break

                data = decode_stream_message(message)
                msg_type = data.get("type")
                self._record_stream_rx(
                    msg_type=msg_type,
                    payload_size=self._payload_size_bytes(message),
                    step=data.get("step"),
                )

                with self._data_condition:
                    if msg_type == "observation":
                        self._latest_observation = data
                    elif msg_type == "action":
                        self._latest_action = data
                    self._data_condition.notify_all()

                if msg_type == "observation" and self._on_observation_callback is not None:
                    self._on_observation_callback(data)
                elif msg_type == "action" and self._on_action_callback is not None:
                    self._on_action_callback(data)
            except Exception as exc:
                if self._running:
                    if isinstance(exc, ConnectionClosedOK):
                        logger.info("Stream 通道正常关闭")
                    elif isinstance(exc, ConnectionClosed):
                        logger.warning("Stream 通道关闭：%s", exc)
                    else:
                        logger.error("Stream 接收错误：%s", exc)
                break

        self._connected = False

    @staticmethod
    def _payload_size_bytes(payload: str | bytes) -> int:
        """返回 payload 的字节大小。"""

        if isinstance(payload, bytes):
            return len(payload)
        return len(payload.encode("utf-8"))

    def _record_stream_rx(self, msg_type: Any, payload_size: int, step: Any = None) -> None:
        """记录 stream 接收统计并按步数输出频率。"""

        with self._freq_lock:
            if msg_type == "action":
                self._rx_action_count += 1
                self._rx_action_bytes += int(payload_size)
            self._maybe_log_transfer_frequency(step=step)

    def _record_stream_tx(self, msg_type: Any, payload_size: int) -> None:
        """记录 stream 发送统计。"""

        with self._freq_lock:
            if msg_type == "observation":
                self._tx_observation_count += 1
                self._tx_observation_bytes += int(payload_size)

    def _maybe_log_transfer_frequency(self, step: Any) -> None:
        """按步数周期输出 observation/action 传输频率。"""

        try:
            step_int = int(step)
        except (TypeError, ValueError):
            return

        if (step_int + 1) % self.transfer_log_every_steps != 0:
            return

        now = time.monotonic()
        elapsed = now - self._freq_window_start
        if elapsed <= 0:
            return

        tx_hz = self._tx_observation_count / elapsed
        rx_hz = self._rx_action_count / elapsed
        tx_kbs = self._tx_observation_bytes / elapsed / 1024.0
        rx_kbs = self._rx_action_bytes / elapsed / 1024.0

        if self._supports_color:
            logger.info(
                "\033[1;36m传输频率\033[0m | \033[1;32mTX obs\033[0m: %.1f Hz (%.1f KB/s) | \033[1;33mRX action\033[0m: %.1f Hz (%.1f KB/s)",
                tx_hz,
                tx_kbs,
                rx_hz,
                rx_kbs,
            )
        else:
            logger.info(
                "传输频率 | TX obs: %.1f Hz (%.1f KB/s) | RX action: %.1f Hz (%.1f KB/s)",
                tx_hz,
                tx_kbs,
                rx_hz,
                rx_kbs,
            )

        self._freq_window_start = now
        self._tx_observation_count = 0
        self._tx_observation_bytes = 0
        self._rx_action_count = 0
        self._rx_action_bytes = 0

    def _reconnect_control(self) -> None:
        """重连 control 通道。"""

        if self._control_connection is not None:
            try:
                self._control_connection.close()
            except Exception:
                pass

        logger.debug("Control 连接已断开，尝试重连：%s", self.control_uri)
        self._control_connection = connect(
            self.control_uri,
            open_timeout=self.connection_timeout,
            close_timeout=self.connection_timeout,
            max_size=self.websocket_max_size_bytes,
        )
        self._connected = self._stream_connection is not None
        logger.debug("Control 重连成功")

    def _reconnect_stream(self) -> None:
        """重连 stream 通道。"""

        if self._stream_connection is not None:
            try:
                self._stream_connection.close()
            except Exception:
                pass

        logger.debug("Stream 连接已断开，尝试重连：%s", self.stream_uri)
        self._stream_connection = connect(
            self.stream_uri,
            open_timeout=self.connection_timeout,
            close_timeout=self.connection_timeout,
            max_size=self.websocket_max_size_bytes,
        )
        self._connected = self._control_connection is not None
        logger.debug("Stream 重连成功")

    def _send_control_message(self, message: dict[str, Any]) -> None:
        """发送 control 消息。

        Args:
            message: 待发送的控制消息字典。
        """

        if self._control_connection is None:
            raise RuntimeError("未连接到 infer 控制通道")

        payload = json.dumps(message, ensure_ascii=False)
        with self._control_lock:
            try:
                self._control_connection.send(payload)
            except ConnectionClosed as exc:
                logger.debug("Control 连接关闭（%s），准备重连后重试", exc)
                self._reconnect_control()
                self._control_connection.send(payload)

    def _send_stream_message(self, message: dict[str, Any]) -> None:
        """发送 stream 消息。

        Args:
            message: 待发送的 stream 消息字典。
        """

        payload = encode_stream_message(message)
        with self._stream_send_lock:
            if self._stream_connection is None or not self._connected:
                self._reconnect_stream()
            assert self._stream_connection is not None
            try:
                self._stream_connection.send(payload)
            except ConnectionClosed as exc:
                logger.debug("Stream 连接关闭（%s），准备重连后重试", exc)
                self._reconnect_stream()
                assert self._stream_connection is not None
                self._stream_connection.send(payload)

            self._record_stream_tx(msg_type=message.get("type"), payload_size=len(payload))

    def send_start(self, episode_id: int = 0) -> None:
        """发送 start 命令。

        Args:
            episode_id: 即将开始的 episode 编号。
        """

        self._send_control_message(build_control_message("start", episode_id=episode_id))
        logger.debug("发送 start 命令：episode=%s", episode_id)

    def send_reset(self, episode_id: int = 0, data: dict[str, Any] | None = None) -> None:
        """发送 reset 命令。

        Args:
            episode_id: 目标 episode 编号。
            data: 可选附加负载。
        """

        self._send_control_message(
            build_control_message("reset", episode_id=episode_id, data=data)
        )
        logger.debug("发送 reset 命令：episode=%s", episode_id)

    def send_episode_end(
        self,
        episode_id: int,
        status: str,
        reason: str,
        metrics: dict[str, Any] | None,
        step: int,
        need_reset: bool = True,
    ) -> None:
        """发送统一 episode_end 控制消息。

        Args:
            episode_id: 结束的 episode 编号。
            status: 结束状态，例如 success/failed/timeout。
            reason: 结束原因。
            metrics: 评测指标。
            step: 结束时的 step。
            need_reset: 是否要求 infer 侧执行 policy reset。
        """

        message = build_episode_end_message(
            episode_id=episode_id,
            status=status,
            reason=reason,
            metrics=metrics,
            step=step,
            need_reset=need_reset,
        )
        self._send_control_message(message)
        logger.debug("发送 episode_end 命令：episode=%s status=%s", episode_id, status)

    def send_stop(self) -> None:
        """发送 stop 命令。"""

        if not self._connected or self._control_connection is None:
            logger.warning("未连接到 infer，跳过发送 stop 命令")
            return

        self._send_control_message(build_control_message("stop"))
        logger.debug("发送 stop 命令")

    def send_heartbeat(self) -> None:
        """发送 heartbeat 命令。"""

        self._send_control_message(build_control_message("heartbeat"))

    def send_observation(
        self,
        observation: dict[str, Any],
        task: str,
        episode_id: int,
        step: int,
        timestamp: float | None = None,
    ) -> None:
        """向 infer 发送完整 observation。

        Args:
            observation: 完整 obs 字典。
            task: 任务名。
            episode_id: 当前 episode 编号。
            step: 当前 step 编号。
            timestamp: 可选时间戳。
        """

        message = build_stream_message(
            msg_type="observation",
            observation=observation,
            task=task,
            episode_id=episode_id,
            step=step,
            timestamp=timestamp,
        )
        self._send_stream_message(message)

    def get_latest_action(self) -> dict[str, Any] | None:
        """获取最新收到的 action 消息。"""

        with self._data_lock:
            return self._latest_action

    def get_latest_observation(self) -> dict[str, Any] | None:
        """获取最新收到的 observation 消息。"""

        with self._data_lock:
            return self._latest_observation

    def wait_for_action(
        self,
        episode_id: int,
        step: int,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """等待指定 episode/step 对应的 action 消息。

        Args:
            episode_id: 目标 episode 编号。
            step: 目标 step 编号。
            timeout: 最长等待时间，单位秒。

        Returns:
            dict[str, Any] | None: 匹配的 action 消息，超时返回 None。
        """

        effective_timeout = self.heartbeat_interval if timeout is None else float(timeout)
        deadline = time.time() + effective_timeout
        with self._data_condition:
            while time.time() < deadline:
                action = self._latest_action
                if (
                    action is not None
                    and int(action.get("episode_id", -1)) == int(episode_id)
                    and int(action.get("step", -1)) == int(step)
                ):
                    return action

                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._data_condition.wait(timeout=remaining)
        return None


def create_client(
    sim_infer_host: str = "sim-infer",
    control_port: int = DEFAULT_WEBSOCKET_CONTROL_PORT,
    stream_port: int = DEFAULT_WEBSOCKET_STREAM_PORT,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT,
    websocket_max_size_bytes: int | None = DEFAULT_WEBSOCKET_MAX_SIZE_BYTES,
) -> EvalWebSocketClient:
    """创建 `EvalWebSocketClient` 便捷函数。

    Args:
        sim_infer_host: infer 容器主机名。
        control_port: control 端口。
        stream_port: stream 端口。

    Returns:
        EvalWebSocketClient: 新建的客户端实例。
    """

    return EvalWebSocketClient(
        sim_infer_host=sim_infer_host,
        control_port=control_port,
        stream_port=stream_port,
        heartbeat_interval=heartbeat_interval,
        connection_timeout=connection_timeout,
        websocket_max_size_bytes=websocket_max_size_bytes,
    )
