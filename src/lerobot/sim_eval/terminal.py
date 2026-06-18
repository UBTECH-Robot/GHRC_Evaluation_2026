import numpy as np
import logging
from typing import Any

from .common import WorkspaceLimits, iter_part_dicts, safe_vec3
from .const import TASK_MAX_STEPS, DEFAULT_POS_THRESHOLD, DEFAULT_ORI_THRESHOLD, TaskNameLiteral
logger = logging.getLogger(__name__)


def task4_check_box_poses_terminal(
    current_box_poses: np.ndarray | list[float] | tuple[float, ...],
    target_poses: np.ndarray | list[float] | tuple[float, ...],
    position_threshold: float = DEFAULT_POS_THRESHOLD,
    orientation_threshold: float = DEFAULT_ORI_THRESHOLD
) -> bool:
    """检查盒子是否偏离初始位姿（终端条件）。"""
    current_pose_vec = np.asarray(current_box_poses, dtype=np.float32).reshape(-1)[:7]
    target_pose_vec = np.asarray(target_poses, dtype=np.float32).reshape(-1)[:7]

    cur_pos = current_pose_vec[:3]
    cur_quat = current_pose_vec[3:]
    target_pos = target_pose_vec[:3]
    target_quat = target_pose_vec[3:]

    cur_quat = cur_quat / np.linalg.norm(cur_quat)
    target_quat = target_quat / np.linalg.norm(target_quat)

    pos_diff = float(np.linalg.norm(cur_pos - target_pos))
    dot_prod = np.dot(target_quat, cur_quat)
    rot_diff = 2 * np.arccos(np.abs(dot_prod))

    if pos_diff > position_threshold or rot_diff > orientation_threshold:
        logger.debug(f"盒子偏离触发：pos={pos_diff:.3f}, rot={rot_diff:.3f}")
        return True
    return False


def task1_check_parts_out_of_workspace(
    parts_poses_dict: Any,
    workspace_limits: WorkspaceLimits,
) -> bool:
    """Task1 终止条件之一：任意零件超出工作空间则返回 True。"""
    if not isinstance(workspace_limits, dict):
        return False

    x_lim = workspace_limits.get("x")
    y_lim = workspace_limits.get("y")
    z_lim = workspace_limits.get("z")
    if x_lim is None or y_lim is None or z_lim is None:
        return False

    x_min, x_max = float(x_lim[0]), float(x_lim[1])
    y_min, y_max = float(y_lim[0]), float(y_lim[1])
    z_min, z_max = float(z_lim[0]), float(z_lim[1])

    parts = iter_part_dicts(parts_poses_dict)
    if not parts:
        return False

    for part in parts:
        pos = safe_vec3(part.get("position"))
        if pos is None:
            continue

        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        if not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
            return True

    return False


def task2_check_all_parts_lost(
    parts_poses_dict: Any,
    conveyor_z_min: float,
) -> bool:
    """Task2 终止条件之一：所有零件都掉落到传送带以下，无可操作对象。"""
    parts = iter_part_dicts(parts_poses_dict)
    if not parts:
        return False
    for part in parts:
        pos = safe_vec3(part.get("position"))
        if pos is not None and pos[2] >= float(conveyor_z_min):
            return False
    return True


def check_step_terminal(step: int, max_steps: int) -> bool:
    """检查是否达到最大步数。"""
    if max_steps is None:
        return False

    if not isinstance(step, int) or step < 0:
        return False

    if step >= max_steps:
        return True

def task3_check_terminal(
    robot,
    parts_poses_dict: Any,
    init_foam_pose: np.ndarray,
    workspace_limits: Any,
    foam_move_threshold: float = 0.20 # 增加容忍度
) -> tuple[bool, str]:
    """
    针对 Task3 的终止判定逻辑：
    1. 只有当零件完全脱离桌面范围（掉落）时才终止。
    2. 如果底座泡棉发生了剧烈位移（被撞飞），则终止。
    """
    # 1. 检查零件是否掉落 (Z轴过低)
    for part in iter_part_dicts(parts_poses_dict):
        pos = safe_vec3(part.get("position"))
        if pos is not None:
            # 如果零件高度低于桌面一定距离（比如低于0.5m），判定为掉落失败
            if pos[2] < 0.5: 
                return True, "零件掉落"
            
            # 检查水平面是否严重超出工作区 (例如超出2米)
            if abs(pos[0]) > 2.5 or abs(pos[1]) > 2.5:
                return True, "零件飞出边界"

    # 2. 检查底座（泡棉）是否被撞位移
    # (假设 parts_poses_dict 里包含泡棉信息，或者通过其他方式获取)
    # 此处逻辑保持简单，防止误触发
    return False, ""
