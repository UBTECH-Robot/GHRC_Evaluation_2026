"""WebSocket 通讯协议定义。

本文件统一定义 `sim-eval` 容器与 `infer` 容器之间的消息格式、序列化辅助函数，
以及双容器评测方案需要的控制命令。

使用示例:

```python
from lerobot.sim_eval.protocol import (
    build_stream_message,
    build_episode_end_message,
    get_observation_from_message,
)

obs_message = build_stream_message(
    msg_type="observation",
    observation={
        "observation.state": [0.0] * 20,
        "observation.images.head_left": [[[0.0]] * 3],
    },
    task="task4",
    episode_id=0,
    step=0,
)

episode_end = build_episode_end_message(
    episode_id=0,
    status="success",
    reason="盒子关节达到目标",
    metrics={"task4_total_score": 100},
    step=42,
)

obs = get_observation_from_message(obs_message)
```
"""

from __future__ import annotations

import io
import json
import os
import pickle
import time
import zlib
from dataclasses import asdict, dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import lz4.frame
import msgpack
import numpy as np
from PIL import Image


ControlMessageLiteral = Literal["start", "reset", "stop", "heartbeat", "episode_end"]
EpisodeStatusLiteral = Literal["success", "failed", "timeout", "error", "rerecord", "reset"]


class MessageType(str, Enum):
    """双容器架构使用的消息类型。"""

    OBSERVATION = "observation"
    ACTION = "action"
    START = "start"
    RESET = "reset"
    STOP = "stop"
    HEARTBEAT = "heartbeat"
    EPISODE_END = "episode_end"
    ACK = "ack"
    ERROR = "error"


@dataclass
class ControlMessage:
    """控制命令消息。

    Attributes:
        type: 控制命令类型。
        episode_id: 当前 episode 编号。
        step: 当前 step 编号。
        status: episode 结束状态。
        reason: 结束原因或说明。
        metrics: 评测指标。
        need_reset: 是否要求 infer 侧执行 reset。
        data: 额外自定义负载。
    """

    type: ControlMessageLiteral
    episode_id: int | None = None
    step: int | None = None
    status: EpisodeStatusLiteral | None = None
    reason: str | None = None
    metrics: dict[str, Any] | None = None
    need_reset: bool | None = None
    data: dict[str, Any] | None = None

    def to_json(self) -> str:
        """序列化为 JSON 字符串。"""

        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "ControlMessage":
        """从 JSON 字符串反序列化控制消息。"""

        return cls(**parse_message(json_str))


@dataclass
class ResponseMessage:
    """控制命令应答消息。"""

    type: Literal["ack", "error"]
    message: str
    success: bool

    def to_json(self) -> str:
        """序列化为 JSON 字符串。"""

        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "ResponseMessage":
        """从 JSON 字符串反序列化响应消息。"""

        return cls(**parse_message(json_str))


DEFAULT_WEBSOCKET_SERVER_HOST = "0.0.0.0"
DEFAULT_WEBSOCKET_CONTROL_PORT = 8765
DEFAULT_WEBSOCKET_STREAM_PORT = 8766
DEFAULT_HEARTBEAT_INTERVAL = 5.0
DEFAULT_CONNECTION_TIMEOUT = 30.0
DEFAULT_WEBSOCKET_MAX_SIZE_BYTES = 32 * 1024 * 1024
STREAM_BINARY_MAGIC = b"SIMWS3"
LEGACY_STREAM_BINARY_MAGIC = b"SIMWS2"

OBS_STATE_KEY = "observation.state"
OBS_ENV_KEY = "observation.environment_state"

PACK_KIND_FLOAT32 = "f32"
PACK_KIND_NDARRAY = "nd"
PACK_KIND_JPEG = "jpg"

JPEG_QUALITY = 85
JPEG_SUBSAMPLING = 2
LZ4_COMPRESSION_LEVEL = 0
_IMAGE_WORKERS = max(1, min(4, os.cpu_count() or 1))
_IMAGE_ENCODE_EXECUTOR = ThreadPoolExecutor(max_workers=_IMAGE_WORKERS, thread_name_prefix="simws-jpeg")


def make_json_safe(data: Any) -> Any:
    """将消息体递归转换为可 JSON 序列化的原生 Python 类型。"""

    if data is None:
        return None
    if isinstance(data, (str, int, float, bool)):
        return data
    if isinstance(data, dict):
        return {str(key): make_json_safe(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [make_json_safe(item) for item in data]
    if hasattr(data, "detach") and callable(data.detach):
        try:
            return make_json_safe(data.detach().cpu().tolist())
        except Exception:
            pass
    if hasattr(data, "tolist") and callable(data.tolist):
        try:
            return make_json_safe(data.tolist())
        except Exception:
            pass
    return data


def _to_numpy_array(value: Any) -> np.ndarray:
    """将输入统一转换为 ndarray。"""

    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and callable(value.detach):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


def _pack_float32_array(array: np.ndarray) -> dict[str, Any]:
    """将数组编码为 float32 紧凑二进制。"""

    arr = np.asarray(array, dtype=np.float32)
    return {
        "kind": PACK_KIND_FLOAT32,
        "shape": list(arr.shape),
        "data": arr.tobytes(order="C"),
    }


def _pack_ndarray(array: np.ndarray) -> dict[str, Any]:
    """将任意 ndarray 编码为紧凑二进制。"""

    arr = np.asarray(array)
    return {
        "kind": PACK_KIND_NDARRAY,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": arr.tobytes(order="C"),
    }


def _to_uint8_hwc_rgb(image_value: Any) -> np.ndarray:
    """将图像转换为 HWC RGB uint8。"""

    image = _to_numpy_array(image_value)
    if image.dtype.kind == "f":
        image = np.clip(image * 255.0, 0.0, 255.0)
    image = image.astype(np.uint8, copy=False)

    if image.ndim != 3:
        raise ValueError(f"图像维度必须为3，收到 shape={image.shape}")

    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    elif image.shape[-1] != 3:
        raise ValueError(f"图像通道必须为3，收到 shape={image.shape}")
    return np.ascontiguousarray(image)


def _pack_jpeg_image(image_value: Any) -> dict[str, Any]:
    """将图像编码为 JPEG 二进制并记录目标 CHW 形状。"""

    image_hwc_rgb = _to_uint8_hwc_rgb(image_value)
    height, width = image_hwc_rgb.shape[:2]

    with io.BytesIO() as buffer:
        Image.fromarray(image_hwc_rgb, mode="RGB").save(
            buffer,
            format="JPEG",
            quality=JPEG_QUALITY,
            subsampling=JPEG_SUBSAMPLING,
            optimize=False,
        )
        encoded = buffer.getvalue()

    return {
        "kind": PACK_KIND_JPEG,
        "shape": [3, int(height), int(width)],
        "data": encoded,
    }


def _normalize_observation_for_stream(observation: dict[str, Any]) -> dict[str, Any]:
    """将 observation 归一化为适合二进制传输的格式。"""

    normalized: dict[str, Any] = {}
    image_items: list[tuple[str, Any]] = []

    for key, value in observation.items():
        key_str = str(key)
        if key_str.startswith("observation.images."):
            image_items.append((key_str, value))
            continue

        if key_str in (OBS_STATE_KEY, OBS_ENV_KEY):
            normalized[key_str] = _pack_float32_array(_to_numpy_array(value))
            continue

        if isinstance(value, (str, int, float, bool)):
            normalized[key_str] = value
            continue

        array_value = _to_numpy_array(value)
        normalized[key_str] = _pack_ndarray(array_value)

    if image_items:
        futures = [
            _IMAGE_ENCODE_EXECUTOR.submit(_pack_jpeg_image, image_value)
            for _, image_value in image_items
        ]
        for (key_str, _), future in zip(image_items, futures, strict=True):
            normalized[key_str] = future.result()

    return normalized


def build_stream_message(
    msg_type: str,
    observation: dict[str, Any] | None = None,
    action: list[float] | Any | None = None,
    task: str = "",
    episode_id: int = 0,
    step: int = 0,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """构建 stream 消息。

    Args:
        msg_type: `"observation"` 或 `"action"`。
        observation: 完整 observation 字典。
        action: 动作向量。
        task: 任务名。
        episode_id: episode 编号。
        step: step 编号。
        timestamp: 时间戳，默认使用当前时间。

    Returns:
        dict[str, Any]: 可直接序列化的消息字典。
    """

    if timestamp is None:
        timestamp = time.time()

    normalized_observation: dict[str, Any] = {}
    if observation is not None:
        normalized_observation = _normalize_observation_for_stream(observation)

    message = {
        "type": msg_type,
        "observation": normalized_observation,
        "task": task,
        "episode_id": int(episode_id),
        "step": int(step),
        "timestamp": float(timestamp),
    }
    if action is not None:
        action_array = _to_numpy_array(action).reshape(-1)
        message["action"] = _pack_float32_array(action_array)
    return message


def build_control_message(
    control_type: ControlMessageLiteral,
    episode_id: int | None = None,
    data: dict[str, Any] | None = None,
    step: int | None = None,
    status: EpisodeStatusLiteral | None = None,
    reason: str | None = None,
    metrics: dict[str, Any] | None = None,
    need_reset: bool | None = None,
) -> dict[str, Any]:
    """构建 control 消息。"""

    message: dict[str, Any] = {"type": control_type}
    if episode_id is not None:
        message["episode_id"] = int(episode_id)
    if step is not None:
        message["step"] = int(step)
    if status is not None:
        message["status"] = status
    if reason is not None:
        message["reason"] = reason
    if metrics is not None:
        message["metrics"] = make_json_safe(metrics)
    if need_reset is not None:
        message["need_reset"] = bool(need_reset)
    if data is not None:
        message["data"] = make_json_safe(data)
    return message


def build_episode_end_message(
    episode_id: int,
    status: EpisodeStatusLiteral,
    reason: str,
    metrics: dict[str, Any] | None,
    step: int,
    need_reset: bool = True,
) -> dict[str, Any]:
    """构建统一的 episode_end 控制消息。"""

    return build_control_message(
        control_type=MessageType.EPISODE_END.value,
        episode_id=episode_id,
        status=status,
        reason=reason,
        metrics=metrics,
        step=step,
        need_reset=need_reset,
    )


def get_observation_from_message(message: dict[str, Any]) -> dict[str, Any]:
    """从 stream 消息中恢复完整 observation 字典。"""

    observation = message.get("observation")
    if isinstance(observation, dict) and observation:
        decoded: dict[str, Any] = {}
        for key, value in observation.items():
            if not isinstance(value, dict):
                decoded[key] = value
                continue

            kind = value.get("kind")
            data = value.get("data")
            shape = value.get("shape")

            if kind == PACK_KIND_JPEG and isinstance(data, (bytes, bytearray)):
                with io.BytesIO(bytes(data)) as buffer:
                    image_hwc = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)
                image_chw = np.transpose(image_hwc, (2, 0, 1)).copy()
                if isinstance(shape, list) and len(shape) == 3:
                    target_shape = tuple(int(x) for x in shape)
                    if image_chw.shape != target_shape:
                        image_chw = image_chw.reshape(target_shape)
                decoded[key] = image_chw
                continue

            if kind == PACK_KIND_FLOAT32 and isinstance(data, (bytes, bytearray)) and isinstance(shape, list):
                decoded[key] = np.frombuffer(bytes(data), dtype=np.float32).reshape(tuple(int(x) for x in shape)).copy()
                continue

            if (
                kind == PACK_KIND_NDARRAY
                and isinstance(data, (bytes, bytearray))
                and isinstance(shape, list)
                and isinstance(value.get("dtype"), str)
            ):
                decoded[key] = (
                    np.frombuffer(bytes(data), dtype=np.dtype(value["dtype"]))
                    .reshape(tuple(int(x) for x in shape))
                    .copy()
                )
                continue

            decoded[key] = value

        return decoded
    return {}


def parse_message(raw_message: str) -> dict[str, Any]:
    """解析 JSON 消息字符串。"""

    try:
        data = json.loads(raw_message)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败：{exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"消息必须是字典格式，收到：{type(data)}")
    return data


def _msgpack_default(data: Any) -> Any:
    """msgpack 序列化默认转换。"""

    if isinstance(data, np.ndarray):
        return _pack_ndarray(data)
    if isinstance(data, (np.integer,)):
        return int(data)
    if isinstance(data, (np.floating,)):
        return float(data)
    raise TypeError(f"不支持的 msgpack 序列化类型：{type(data)}")


def encode_stream_message(message: dict[str, Any]) -> bytes:
    """编码 stream 消息为二进制压缩帧。"""

    raw = msgpack.packb(message, use_bin_type=True, default=_msgpack_default)
    compressed = lz4.frame.compress(raw, compression_level=LZ4_COMPRESSION_LEVEL)
    return STREAM_BINARY_MAGIC + compressed


def decode_stream_message(raw_message: str | bytes) -> dict[str, Any]:
    """解码 stream 消息，兼容历史 JSON 文本。"""

    if isinstance(raw_message, (bytes, bytearray)):
        raw_bytes = bytes(raw_message)
        if raw_bytes.startswith(STREAM_BINARY_MAGIC):
            compressed = raw_bytes[len(STREAM_BINARY_MAGIC) :]
            data = msgpack.unpackb(
                lz4.frame.decompress(compressed),
                raw=False,
                strict_map_key=False,
            )
            if not isinstance(data, dict):
                raise ValueError(f"二进制消息必须是字典格式，收到：{type(data)}")
            action = data.get("action")
            if isinstance(action, dict) and action.get("kind") == PACK_KIND_FLOAT32:
                action_data = action.get("data")
                action_shape = action.get("shape")
                if isinstance(action_data, (bytes, bytearray)) and isinstance(action_shape, list):
                    data["action"] = np.frombuffer(
                        bytes(action_data),
                        dtype=np.float32,
                    ).reshape(tuple(int(x) for x in action_shape)).copy()
            return data
        if raw_bytes.startswith(LEGACY_STREAM_BINARY_MAGIC):
            compressed = raw_bytes[len(LEGACY_STREAM_BINARY_MAGIC) :]
            data = pickle.loads(zlib.decompress(compressed))
            if not isinstance(data, dict):
                raise ValueError(f"二进制消息必须是字典格式，收到：{type(data)}")
            return data
        return parse_message(raw_bytes.decode("utf-8"))
    return parse_message(raw_message)
