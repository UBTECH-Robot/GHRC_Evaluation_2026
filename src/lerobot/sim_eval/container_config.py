"""容器脚本配置加载工具。

本文件为 `sim-eval` 与 `infer` 两个新入口统一提供 YAML 配置读取、键名规范化、
相对路径解析与基础校验能力。

使用示例:

```python
from lerobot.sim_eval.container_config import (
    load_container_config,
    require_config_keys,
)

config = load_container_config(
    config_path="eval_config/task4_sim_eval.yaml",
    defaults={"device": "cuda:0"},
    path_fields={"task_config_path", "log_dir", "policy_path"},
)
require_config_keys(config, ["task", "task_config_path"], "sim-eval")
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import yaml


def _normalize_key(key: str) -> str:
    """将 YAML 键名统一转成 snake_case 近似形式。"""

    return key.replace("-", "_").strip()


def load_container_config(
    config_path: str | Path,
    defaults: dict[str, Any] | None = None,
    path_fields: set[str] | None = None,
) -> argparse.Namespace:
    """从 YAML 读取容器配置并返回 `argparse.Namespace`。

    Args:
        config_path: YAML 配置文件路径。
        defaults: 默认配置字典。
        path_fields: 需要按配置文件相对路径解析的字段名集合。

    Returns:
        argparse.Namespace: 归一化后的配置对象。
    """

    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_file}")

    with open(config_file, "r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"配置文件顶层必须是字典：{config_file}")

    normalized: dict[str, Any] = {}
    for key, value in raw_config.items():
        if not isinstance(key, str):
            raise ValueError(f"配置键必须是字符串，收到：{type(key)}")
        normalized[_normalize_key(key)] = value

    merged = dict(defaults or {})
    merged.update(normalized)

    for field in path_fields or set():
        value = merged.get(field)
        if isinstance(value, str) and value:
            candidate = Path(value).expanduser()
            merged[field] = str(candidate if candidate.is_absolute() else (config_file.parent / candidate).resolve())

    return argparse.Namespace(config=str(config_file), **merged)


def require_config_keys(
    config: argparse.Namespace,
    required_keys: Iterable[str],
    config_name: str,
) -> None:
    """校验配置必须包含的键。

    Args:
        config: 已加载的配置对象。
        required_keys: 必需字段列表。
        config_name: 配置名，用于错误提示。
    """

    missing = [key for key in required_keys if getattr(config, key, None) in (None, "")]
    if missing:
        raise ValueError(f"{config_name} 缺少必要配置项：{missing}")
