"""任务评测配置读取工具。

本文件负责从 `Ubtech_sim/config` 下的任务 YAML 中提取评测相关参数，
确保 `eval_config` 只保留多容器隔离运行需要的配置项。

使用示例:

```python
from lerobot.sim_eval.task_eval_config import (
    load_task_yaml_config,
    build_assertion_args_from_task_yaml,
)

task_config = load_task_yaml_config("Ubtech_sim/config/Packing_Box.yaml")
assertion_args = build_assertion_args_from_task_yaml("task4", task_config)
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .common import build_workspace_limits


def load_task_yaml_config(task_config_path: str | Path) -> dict[str, Any]:
    """读取任务 YAML 配置。

    Args:
        task_config_path: 任务 YAML 文件路径。

    Returns:
        dict[str, Any]: YAML 顶层字典。
    """

    config_path = Path(task_config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"任务配置文件不存在：{config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        task_config = yaml.safe_load(file) or {}
    if not isinstance(task_config, dict):
        raise ValueError(f"任务配置顶层必须是字典：{config_path}")
    return task_config


def build_assertion_args_from_task_yaml(
    task: str,
    task_config: dict[str, Any],
    base_args: argparse.Namespace | None = None,
) -> argparse.Namespace:
    """从任务 YAML 生成 `create_task_assertion` 所需参数。

    Args:
        task: 任务名，例如 `task1`。
        task_config: 任务 YAML 内容。
        base_args: 可选基础参数，会先复制到返回对象中。

    Returns:
        argparse.Namespace: 含断言所需字段的参数对象。
    """

    normalized_task = str(task).lower()
    args_dict = dict(vars(base_args)) if base_args is not None else {}
    evaluation = task_config.get("evaluation", {}) or {}
    if not isinstance(evaluation, dict):
        raise ValueError("任务配置中的 evaluation 字段必须是字典")

    if normalized_task == "task1":
        args_dict.update(_build_task1_args(task_config, evaluation))
    elif normalized_task == "task2":
        args_dict.update(_build_task2_args(task_config, evaluation))
    elif normalized_task == "task3":
        args_dict.update(_build_task3_args(task_config, evaluation))
    elif normalized_task == "task4":
        args_dict.update(_build_task4_args(task_config, evaluation))
    else:
        raise ValueError(f"不支持的任务类型：{task}")

    return argparse.Namespace(**args_dict)


def _build_task1_args(task_config: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    """从任务 YAML 构建 Task1 断言参数（官方标准）。

    官方默认值：
    - 抓取/放置: 每零件 10 分，最多 4 个，满分 40+40。
    - 时间: 满分 20，180 秒内完成，每超 30 秒扣 5 分。
    - 成功门槛: 80 分。
    - 抬升判定: 相对初始高度提升 0.10 米。
    """
    box_scale = _get_first_vector(task_config.get("box", {}).get("box_scale"), [0.38, 0.75, 0.36])
    return {
        "task1_lift_height": float(evaluation.get("lift_height", task_config.get("grasp", {}).get("lift_height", 0.17))),
        "task1_workspace_limits": evaluation.get("workspace_limits", build_workspace_limits(task_config)),
        "task1_box_half_size": evaluation.get("box_half_size", [box_scale[0] / 2.0, box_scale[1] / 2.0, box_scale[2] / 2.0]),
        "task1_lift_score_per_part": int(evaluation.get("lift_score_per_part", 10)),
        "task1_box_score_per_part": int(evaluation.get("box_score_per_part", 10)),
        "task1_max_parts": int(evaluation.get("max_parts", task_config.get("part", {}).get("num_parts", 2) * 2)),
        "task1_time_full_score": int(evaluation.get("time_full_score", 20)),
        "task1_time_full_time_seconds": float(evaluation.get("time_full_time_seconds", 180.0)),
        "task1_time_penalty_interval_seconds": float(evaluation.get("time_penalty_interval_seconds", 30.0)),
        "task1_time_penalty_per_interval": int(evaluation.get("time_penalty_per_interval", 5)),
        "task1_success_score_threshold": int(evaluation.get("success_score_threshold", 80)),
        "task1_parts_movement_threshold": float(evaluation.get("parts_movement_threshold", 0.01)),
        "task1_lift_delta": float(evaluation.get("lift_delta", 0.10)),
    }


def _build_task2_args(task_config: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    """从任务 YAML 构建 Task2 断言参数（官方标准）。

    官方默认值：
    - 跟随: 每零件 2.5 分，端执行器 10 cm 范围内计分，最多 8 个，满分 20。
    - 抓取: 每零件 5 分，相对初始高度抬升 0.10 米计分，满分 40。
    - 分拣: 每零件 5 分，正确料箱内释放静止后计分，满分 40。
    - 成功门槛: 100 分。
    """
    box_scale = _get_first_vector(task_config.get("box", {}).get("box_scale"), [0.38, 0.75, 0.36])
    scatter_area = task_config.get("grasp", {}).get("scatter_area", {}) or {}
    center = _coerce_vector(scatter_area.get("center"), [0.12, 0.26859, 1.2], 3)
    size = _coerce_vector(scatter_area.get("size"), [0.06, 0.04], 2)
    plane_position = _get_first_vector(task_config.get("plane", {}).get("plane_position"), center)
    default_conveyor_limits = {
        "x": [center[0] - size[0], center[0] + size[0]],
        "y": [center[1] - size[1], center[1] + size[1]],
        "z": [plane_position[2] - 0.1, plane_position[2] + 0.1],
    }
    conveyor_belt_position = _get_first_vector(
        task_config.get("ConveyorBelt", {}).get("ConveyorBelt_position"),
        plane_position,
    )
    max_parts = int(evaluation.get("max_parts", task_config.get("part", {}).get("num_parts", 4) * 2))
    grab_score_per_part = float(evaluation.get("grab_score_per_part", 5.0))
    sort_score_per_part = float(evaluation.get("sort_score_per_part", 5.0))
    follow_score_per_part = float(evaluation.get("follow_score_per_part", 2.5))
    return {
        "task2_conveyor_limits": evaluation.get("conveyor_limits", default_conveyor_limits),
        "task2_conveyor_drop_z": float(evaluation.get("conveyor_drop_z", conveyor_belt_position[2])),
        "task2_bin_half_size": evaluation.get("bin_half_size", [box_scale[0] / 2.0, box_scale[1] / 2.0, box_scale[2] / 2.0]),
        "task2_grab_score_per_part": grab_score_per_part,
        "task2_sort_score_per_part": sort_score_per_part,
        "task2_max_parts": max_parts,
        "task2_success_score_threshold": int(evaluation.get("success_score_threshold", 100)),
        "task2_follow_score_per_part": follow_score_per_part,
        "task2_follow_distance_threshold": float(evaluation.get("follow_distance_threshold", 0.10)),
        "task2_lift_delta": float(evaluation.get("lift_delta", 0.10)),
    }


def _build_task3_args(task_config: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    """从任务 YAML 构建 Task3 断言参数（官方标准）。

    官方默认值：
    - 抓取: 每零件 7.5 分，相对初始高度抬升 0.10 米计分，最多 6 个，满分 45。
    - 嵌装: 每零件 7.5 分，匹配槽位内释放静止后计分，满分 45。
    - 时间: 满分 10，360 秒内完成，每超 60 秒扣 5 分。
    - 成功门槛: 90 分。
    """
    box_positions = task_config.get("box", {}).get("box_position", []) or []
    box_scales = task_config.get("box", {}).get("box_scale", []) or []
    default_workspace_limits = _build_task3_workspace_limits(box_positions, box_scales)
    return {
        "task3_foam_pos": evaluation.get("foam_pos", task_config.get("foam", {}).get("foam_position", [0.76, 0.3, 1.0])),
        "task3_workspace_limits": evaluation.get("workspace_limits", default_workspace_limits),
        "task3_dist_threshold": float(evaluation.get("dist_threshold", 0.05)),
        "task3_height_threshold": float(evaluation.get("height_threshold", 0.1)),
        "task3_success_score_threshold": int(evaluation.get("success_score_threshold", 90)),
        "task3_grab_score_per_part": float(evaluation.get("grab_score_per_part", 7.5)),
        "task3_insert_score_per_part": float(evaluation.get("insert_score_per_part", 7.5)),
        "task3_max_parts": int(evaluation.get("max_parts", task_config.get("part", {}).get("num_parts", 3) * 2)),
        "task3_lift_delta": float(evaluation.get("lift_delta", 0.10)),
        "task3_time_full_score": int(evaluation.get("time_full_score", 10)),
        "task3_time_full_time_seconds": float(evaluation.get("time_full_time_seconds", 360.0)),
        "task3_time_penalty_interval_seconds": float(evaluation.get("time_penalty_interval_seconds", 60.0)),
        "task3_time_penalty_per_interval": int(evaluation.get("time_penalty_per_interval", 5)),
    }


def _build_task4_args(task_config: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    """从任务 YAML 构建 Task4 断言参数（官方标准）。

    官方默认值：
    - 短边: 每边 15 分，需端执行器接触后闭合，满分 30。
    - 长边: 每边 15 分，需端执行器接触后闭合，满分 30。
    - 时间: 满分 10，180 秒内完成，每超 30 秒扣 5 分。
    - 已移除官方不存在的协作系数（single_arm_factor / bimanual_factor）。
    """
    del task_config
    return {
        "task4_short_targets": evaluation.get("short_targets", [-3.3219733, -3.3213105]),
        "task4_long_targets": evaluation.get("long_targets", [-3.4906585, -3.4906585]),
        "task4_short_edge_joint_indices": evaluation.get("short_edge_joint_indices", [2, 3]),
        "task4_long_edge_joint_indices": evaluation.get("long_edge_joint_indices", [0, 1]),
        "task4_joint_threshold": float(evaluation.get("joint_threshold", 0.2)),
        "task4_action_movement_eps": float(evaluation.get("action_movement_eps", 0.001)),
        "task4_co_move_ratio_threshold": float(evaluation.get("co_move_ratio_threshold", 0.1)),
        "task4_success_hold_steps": int(evaluation.get("success_hold_steps", 10)),
        "task4_short_edge_score_per_edge": int(evaluation.get("short_edge_score_per_edge", 15)),
        "task4_long_edge_score_per_edge": int(evaluation.get("long_edge_score_per_edge", 15)),
        "task4_time_full_score": int(evaluation.get("time_full_score", 10)),
        "task4_time_full_time_seconds": float(evaluation.get("time_full_time_seconds", 180.0)),
        "task4_time_penalty_interval_seconds": float(evaluation.get("time_penalty_interval_seconds", 30.0)),
        "task4_time_penalty_per_interval": int(evaluation.get("time_penalty_per_interval", 5)),
        "task4_contact_distance_threshold": float(evaluation.get("contact_distance_threshold", 0.05)),
        "task4_box_pose_position_threshold": float(evaluation.get("box_pose_position_threshold", 1.0)),
        "task4_box_pose_orientation_threshold": float(evaluation.get("box_pose_orientation_threshold", 1.0)),
    }


def _get_first_vector(value: Any, default: list[float]) -> list[float]:
    """从可能是二维列表的 YAML 值中取第一组向量。"""

    if isinstance(value, list) and value and isinstance(value[0], list):
        return [float(item) for item in value[0]]
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(item) for item in default]


def _coerce_vector(value: Any, default: list[float], expected_length: int) -> list[float]:
    """将输入规范化为固定长度的浮点向量。"""

    if not isinstance(value, list) or len(value) < expected_length:
        return [float(item) for item in default]
    return [float(value[index]) for index in range(expected_length)]


def _build_task3_workspace_limits(box_positions: Any, box_scales: Any) -> dict[str, list[float]]:
    """从 task3 的箱体配置推导工作空间边界。"""

    default_limits = {
        "x": [0.0, 1.6],
        "y": [-0.1, 0.7],
        "z": [0.8, 1.3],
    }
    if not isinstance(box_positions, list) or not box_positions:
        return default_limits

    x_mins: list[float] = []
    x_maxs: list[float] = []
    y_mins: list[float] = []
    y_maxs: list[float] = []
    z_mins: list[float] = []
    z_maxs: list[float] = []

    for index, position in enumerate(box_positions):
        if not isinstance(position, list) or len(position) < 3:
            continue
        scale = box_scales[index] if isinstance(box_scales, list) and index < len(box_scales) else [0.5, 1.2, 0.3]
        if not isinstance(scale, list) or len(scale) < 3:
            scale = [0.5, 1.2, 0.3]
        half_x = float(scale[0]) / 2.0
        half_y = float(scale[1]) / 2.0
        half_z = float(scale[2]) / 2.0
        x_mins.append(float(position[0]) - half_x)
        x_maxs.append(float(position[0]) + half_x)
        y_mins.append(float(position[1]) - half_y)
        y_maxs.append(float(position[1]) + half_y)
        z_mins.append(float(position[2]) - half_z)
        z_maxs.append(float(position[2]) + half_z)

    if not x_mins:
        return default_limits

    return {
        "x": [min(x_mins), max(x_maxs)],
        "y": [min(y_mins), max(y_maxs)],
        "z": [min(z_mins), max(z_maxs) + 0.2],
    }
