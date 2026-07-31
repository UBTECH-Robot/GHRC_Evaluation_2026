"""WebSocket Server。

本文件用于 `infer` 容器启动同步 WebSocket 服务，接收 `sim-eval` 侧控制命令、
读取完整 observation stream，并回传 action 数据。

使用示例:

```python
from lerobot.sim_eval.websocket_server import SimInferWebSocketServer

server = SimInferWebSocketServer(host="0.0.0.0", control_port=8765, stream_port=8766)
server.start()
server.broadcast_action([0.0] * 20, episode_id=0, step=0)
server.stop()
```
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from typing import Any, Callable

try:
    _ws_sync_server = importlib.import_module("websockets.sync.server")
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "未找到 websockets.sync.server，请安装 websockets>=11：pip install -U websockets"
    ) from exc

ServerConnection = _ws_sync_server.ServerConnection
serve = _ws_sync_server.serve
_ws_exceptions = importlib.import_module("websockets.exceptions")
ConnectionClosed = _ws_exceptions.ConnectionClosed
ConnectionClosedOK = getattr(_ws_exceptions, "ConnectionClosedOK", ConnectionClosed)

from .protocol import (
    DEFAULT_CONNECTION_TIMEOUT,
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_WEBSOCKET_MAX_SIZE_BYTES,
    MessageType,
    ResponseMessage,
    DEFAULT_WEBSOCKET_CONTROL_PORT,
    DEFAULT_WEBSOCKET_SERVER_HOST,
    DEFAULT_WEBSOCKET_STREAM_PORT,
    build_stream_message,
    decode_stream_message,
    encode_stream_message,
    parse_message,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SimInferWebSocketServer:
    """`infer` 容器使用的同步 WebSocket Server。

    类属性:
        host: 服务绑定地址。
        control_port: 控制命令端口。
        stream_port: 双向 stream 端口。
        stream_clients: 当前已连接的 sim-eval stream client 列表。
    """

    def __init__(
        self,
        host: str = DEFAULT_WEBSOCKET_SERVER_HOST,
        control_port: int = DEFAULT_WEBSOCKET_CONTROL_PORT,
        stream_port: int = DEFAULT_WEBSOCKET_STREAM_PORT,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT,
        websocket_max_size_bytes: int | None = DEFAULT_WEBSOCKET_MAX_SIZE_BYTES,
    ) -> None:
        """初始化 WebSocket 服务器。

        Args:
            host: 服务绑定地址。
            control_port: 控制命令端口。
            stream_port: 双向数据流端口。
        """

        self.host = host
        self.control_port = int(control_port)
        self.stream_port = int(stream_port)
        self.heartbeat_interval = float(heartbeat_interval)
        self.connection_timeout = float(connection_timeout)
        self.websocket_max_size_bytes = (
            None if websocket_max_size_bytes is None else int(websocket_max_size_bytes)
        )

        self.stream_clients: list[ServerConnection] = []
        self._clients_lock = threading.Lock()

        self._latest_action: dict[str, Any] | None = None
        self._latest_observation: dict[str, Any] | None = None
        self._observation_lock = threading.Lock()
        self._observation_condition = threading.Condition(self._observation_lock)
        self._action_lock = threading.Lock()

        self._on_start_callback: Callable[[dict[str, Any]], None] | None = None
        self._on_reset_callback: Callable[[dict[str, Any]], None] | None = None
        self._on_stop_callback: Callable[[], None] | None = None
        self._on_episode_end_callback: Callable[[dict[str, Any]], None] | None = None
        self._on_observation_callback: Callable[[dict[str, Any]], None] | None = None

        self._control_server: Any | None = None
        self._stream_server: Any | None = None
        self._running = False
        self._control_thread: threading.Thread | None = None
        self._stream_thread: threading.Thread | None = None

        self.current_episode_id = 0
        self.current_step = 0

    def set_start_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """设置 start 命令回调。

        Args:
            callback: 处理 start 控制消息的回调。
        """

        self._on_start_callback = callback

    def set_reset_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """设置 reset 命令回调。

        Args:
            callback: 处理 reset 控制消息的回调。
        """

        self._on_reset_callback = callback

    def set_stop_callback(self, callback: Callable[[], None]) -> None:
        """设置 stop 命令回调。

        Args:
            callback: 处理 stop 命令的回调。
        """

        self._on_stop_callback = callback

    def set_episode_end_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """设置 episode_end 命令回调。

        Args:
            callback: 处理 episode_end 控制消息的回调。
        """

        self._on_episode_end_callback = callback

    def set_observation_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """设置 observation stream 回调。

        Args:
            callback: 处理 observation stream 消息的回调。
        """

        self._on_observation_callback = callback

    def register_stream_client(self, ws: ServerConnection) -> None:
        """注册 stream client 连接。

        Args:
            ws: 新建立的 stream 连接对象。
        """

        with self._clients_lock:
            if ws not in self.stream_clients:
                self.stream_clients.append(ws)
                logger.info("Stream client 已连接，当前 clients: %s", len(self.stream_clients))

    def unregister_stream_client(self, ws: ServerConnection) -> None:
        """注销 stream client 连接。

        Args:
            ws: 待移除的 stream 连接对象。
        """

        with self._clients_lock:
            if ws in self.stream_clients:
                self.stream_clients.remove(ws)
                logger.info("Stream client 已断开，当前 clients: %s", len(self.stream_clients))

    def handle_control_message(self, ws: ServerConnection, message: str) -> None:
        """处理 control 通道消息。

        Args:
            ws: 来源连接。
            message: 原始 JSON 消息字符串。
        """

        try:
            data = parse_message(message)
            msg_type = data.get("type")

            if msg_type == MessageType.START.value:
                self.current_episode_id = int(data.get("episode_id", 0))
                self.current_step = 0
                if self._on_start_callback is not None:
                    self._on_start_callback(data)
            elif msg_type == MessageType.RESET.value:
                self.current_episode_id = int(data.get("episode_id", 0))
                self.current_step = 0
                if self._on_reset_callback is not None:
                    self._on_reset_callback(data)
            elif msg_type == MessageType.EPISODE_END.value:
                self.current_episode_id = int(data.get("episode_id", self.current_episode_id))
                self.current_step = int(data.get("step", self.current_step))
                if self._on_episode_end_callback is not None:
                    self._on_episode_end_callback(data)
            elif msg_type == MessageType.STOP.value:
                if self._on_stop_callback is not None:
                    self._on_stop_callback()
            elif msg_type == MessageType.HEARTBEAT.value:
                pass
            else:
                raise ValueError(f"未知的控制消息类型：{msg_type}")

            ws.send(
                ResponseMessage(
                    type="ack",
                    message=f"{msg_type} received",
                    success=True,
                ).to_json()
            )
            logger.info("收到控制命令：%s", msg_type)
        except Exception as exc:
            logger.error("处理 control 消息异常：%s", exc)
            try:
                ws.send(
                    ResponseMessage(
                        type="error",
                        message=str(exc),
                        success=False,
                    ).to_json()
                )
            except Exception:
                logger.debug("发送 error 响应失败，连接可能已关闭")

    def handle_stream_message(self, ws: ServerConnection, message: str | bytes) -> None:
        """处理 stream 通道消息。

        Args:
            ws: 来源连接。
            message: 原始 stream 消息（支持 JSON 文本与二进制帧）。
        """

        del ws
        data = decode_stream_message(message)
        msg_type = data.get("type")

        if msg_type == MessageType.OBSERVATION.value:
            with self._observation_condition:
                self._latest_observation = data
                self.current_episode_id = int(data.get("episode_id", self.current_episode_id))
                self.current_step = int(data.get("step", self.current_step))
                self._observation_condition.notify_all()
            if self._on_observation_callback is not None:
                self._on_observation_callback(data)
            return

        if msg_type == MessageType.ACTION.value:
            with self._action_lock:
                self._latest_action = data
            return

        logger.warning("收到未知 stream 消息类型：%s", msg_type)

    def broadcast_action(
        self,
        action: list[float] | Any,
        episode_id: int | None = None,
        step: int | None = None,
        task: str = "",
    ) -> None:
        """广播动作数据给所有 stream clients。

        Args:
            action: 动作向量。
            episode_id: episode 编号，默认使用当前值。
            step: step 编号，默认使用当前值。
            task: 任务名。
        """

        if episode_id is None:
            episode_id = self.current_episode_id
        if step is None:
            step = self.current_step

        message = build_stream_message(
            msg_type=MessageType.ACTION.value,
            action=action,
            task=task,
            episode_id=episode_id,
            step=step,
        )
        self._broadcast_message(message)
        with self._action_lock:
            self._latest_action = message

    def _broadcast_message(self, message: dict[str, Any]) -> None:
        """内部广播方法。

        Args:
            message: 已构建好的消息字典。
        """

        with self._clients_lock:
            clients = list(self.stream_clients)

        if not clients:
            logger.debug("没有 stream clients，跳过广播")
            return

        try:
            stream_payload = encode_stream_message(message)
        except Exception as exc:
            logger.error("消息序列化失败：%s", exc)
            return

        dead_clients: list[ServerConnection] = []
        for client in clients:
            try:
                client.send(stream_payload)
            except Exception:
                dead_clients.append(client)

        for client in dead_clients:
            self.unregister_stream_client(client)

        logger.debug("已广播消息到 %s 个 clients", len(clients) - len(dead_clients))

    def get_latest_action(self) -> dict[str, Any] | None:
        """获取最新 action 消息。"""

        with self._action_lock:
            return self._latest_action

    def get_latest_observation(self) -> dict[str, Any] | None:
        """获取最新 observation 消息。"""

        with self._observation_lock:
            return self._latest_observation

    def wait_for_observation(
        self,
        episode_id: int | None = None,
        step: int | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any] | None:
        """等待 observation 消息到达。

        Args:
            episode_id: 可选的目标 episode 编号。
            step: 可选的目标 step 编号。
            timeout: 最长等待时间，单位秒。

        Returns:
            dict[str, Any] | None: 匹配 observation，超时返回 None。
        """

        deadline = time.time() + timeout
        with self._observation_condition:
            while time.time() < deadline:
                observation = self._latest_observation
                if observation is not None:
                    episode_match = episode_id is None or int(observation.get("episode_id", -1)) == int(episode_id)
                    step_match = step is None or int(observation.get("step", -1)) == int(step)
                    if episode_match and step_match:
                        return observation
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._observation_condition.wait(timeout=remaining)
        return None

    def _run_control_server(self) -> None:
        """运行 Control 服务器线程。"""

        def handler(ws: ServerConnection) -> None:
            logger.info("Control client 已连接")
            try:
                while self._running:
                    message = ws.recv()
                    if message is None:
                        break
                    self.handle_control_message(ws, message)
            except Exception as exc:
                if self._running:
                    if isinstance(exc, ConnectionClosedOK):
                        logger.info("Control 通道正常关闭")
                    elif isinstance(exc, ConnectionClosed):
                        logger.warning("Control 通道关闭：%s", exc)
                    else:
                        logger.error("Control handler 异常：%s", exc)
            finally:
                logger.info("Control client 已断开")

        self._control_server = serve(
            handler,
            self.host,
            self.control_port,
            open_timeout=self.connection_timeout,
            close_timeout=self.connection_timeout,
            max_size=self.websocket_max_size_bytes,
        )
        logger.info("Control WebSocket 服务器已启动：ws://%s:%s", self.host, self.control_port)
        self._control_server.serve_forever()

    def _run_stream_server(self) -> None:
        """运行 Stream 服务器线程。"""

        def handler(ws: ServerConnection) -> None:
            logger.info("Stream client 已连接")
            self.register_stream_client(ws)
            try:
                while self._running:
                    message = ws.recv()
                    if message is None:
                        break
                    self.handle_stream_message(ws, message)
            except Exception as exc:
                if self._running:
                    if isinstance(exc, ConnectionClosedOK):
                        logger.info("Stream 通道正常关闭")
                    elif isinstance(exc, ConnectionClosed):
                        logger.warning("Stream 通道关闭：%s", exc)
                    else:
                        logger.error("Stream handler 异常并退出：%s", exc)
            finally:
                self.unregister_stream_client(ws)
                logger.info("Stream client 已断开")

        self._stream_server = serve(
            handler,
            self.host,
            self.stream_port,
            open_timeout=self.connection_timeout,
            close_timeout=self.connection_timeout,
            max_size=self.websocket_max_size_bytes,
        )
        logger.info("Stream WebSocket 服务器已启动：ws://%s:%s", self.host, self.stream_port)
        self._stream_server.serve_forever()

    def start(self) -> None:
        """启动 control 与 stream 服务器线程。"""

        self._running = True
        self._control_thread = threading.Thread(target=self._run_control_server, daemon=True)
        self._stream_thread = threading.Thread(target=self._run_stream_server, daemon=True)
        self._control_thread.start()
        self._stream_thread.start()
        logger.info("WebSocket 服务器线程已启动")

    def stop(self) -> None:
        """停止 WebSocket 服务器并清理连接。"""

        self._running = False

        with self._clients_lock:
            for client in self.stream_clients:
                try:
                    client.close()
                except Exception:
                    pass
            self.stream_clients.clear()

        if self._control_server is not None:
            try:
                self._control_server.shutdown()
            except Exception:
                pass

        if self._stream_server is not None:
            try:
                self._stream_server.shutdown()
            except Exception:
                pass

        logger.info("WebSocket 服务器已停止")


def run_server(
    host: str = DEFAULT_WEBSOCKET_SERVER_HOST,
    control_port: int = DEFAULT_WEBSOCKET_CONTROL_PORT,
    stream_port: int = DEFAULT_WEBSOCKET_STREAM_PORT,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT,
    websocket_max_size_bytes: int | None = DEFAULT_WEBSOCKET_MAX_SIZE_BYTES,
) -> SimInferWebSocketServer:
    """启动 `SimInferWebSocketServer` 便捷函数。"""

    server = SimInferWebSocketServer(
        host=host,
        control_port=control_port,
        stream_port=stream_port,
        heartbeat_interval=heartbeat_interval,
        connection_timeout=connection_timeout,
        websocket_max_size_bytes=websocket_max_size_bytes,
    )
    server.start()
    return server


def main() -> None:
    """独立启动 WebSocket 服务器，便于手工调试。"""

    logger.info("启动 WebSocket 服务器...")
    server = run_server()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("收到中断信号，退出...")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
