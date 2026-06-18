from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .common import WalkerS2sim
from .terminal import check_step_terminal, task1_check_parts_out_of_workspace, task2_check_all_parts_lost, task3_check_terminal, task4_check_box_poses_terminal
from .scoring import (
    task1_check_parts_in_box,
    task1_time_out_check,
    task2_check_parts_grabbed,
    task2_check_parts_in_correct_bin,
    task2_calculate_total_score,
    task2_check_end_effector_followed_parts,
    task3_calculate_specific_score,
    task3_check_parts_inserted_in_slots,
    task3_time_score,
    task4_check_box_joints_success,
    task4_check_short_edge_close_score,
    task4_check_long_edge_close_score,
    task4_time_score,
    task4_calculate_total_score,
    calculate_time_score,
    check_parts_lifted_from_initial_height,
)
import logging

logger = logging.getLogger(__name__)


class TaskAssertion:
    """
    断言基类。子类实现 __call__，返回:
        (is_success, terminal, reason, metrics)
    - is_success: 任务是否完成
    - terminal:   是否触发终止（超时或不可恢复的失败）
    """
    def __call__(
        self,
        robot: WalkerS2sim,
        step: int,
        action: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[bool, bool, str, dict]:
        raise NotImplementedError
    
    
class Task4EpisodeActionTracker:
    """记录 episode 内每一步 action，并统计双臂同时运动情况。"""

    def __init__(self):
        self.action_list: list[np.ndarray] = []

    def reset(self) -> None:
        self.action_list.clear()

    def add_action(self, action: np.ndarray | torch.Tensor | None) -> None:
        if action is None:
            return
        if isinstance(action, torch.Tensor):
            arr = action.detach().float().cpu().numpy().reshape(-1)
        else:
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
        self.action_list.append(arr)

    def get_bimanual_collaboration_stats(
        self,
        movement_eps: float = 1e-3,
        co_move_ratio_threshold: float = 0.1,
    ) -> dict[str, Any]:
        """
        通过相邻 step 的 action 变化量判断“同时运动”：
        - 左臂: action[:7]
        - 右臂: action[7:14]
        """
        if len(self.action_list) < 2:
            return {
                "is_bimanual_collaboration": False,
                "co_move_steps": 0,
                "active_steps": 0,
                "co_move_ratio": 0.0,
                "left_only_steps": 0,
                "right_only_steps": 0,
                "valid_action_steps": len(self.action_list),
            }

        co_move_steps = 0
        active_steps = 0
        left_only_steps = 0
        right_only_steps = 0

        for idx in range(1, len(self.action_list)):
            prev = self.action_list[idx - 1]
            cur = self.action_list[idx]
            if prev.size < 14 or cur.size < 14:
                continue

            delta = np.abs(cur[:14] - prev[:14])
            left_move = bool(np.any(delta[:7] > movement_eps))
            right_move = bool(np.any(delta[7:14] > movement_eps))

            if left_move or right_move:
                active_steps += 1
            if left_move and right_move:
                co_move_steps += 1
            elif left_move:
                left_only_steps += 1
            elif right_move:
                right_only_steps += 1

        co_move_ratio = float(co_move_steps / active_steps) if active_steps > 0 else 0.0
        is_bimanual_collaboration = bool(co_move_steps > 0 and co_move_ratio >= co_move_ratio_threshold)

        return {
            "is_bimanual_collaboration": is_bimanual_collaboration,
            "co_move_steps": int(co_move_steps),
            "active_steps": int(active_steps),
            "co_move_ratio": float(co_move_ratio),
            "left_only_steps": int(left_only_steps),
            "right_only_steps": int(right_only_steps),
            "valid_action_steps": int(len(self.action_list)),
        }
        
        
class EpisodePartsTracker:
    """记录 episode 内每一步零件的位姿，跟踪初始高度、静态和释放状态。"""

    def __init__(self):
        self.parts_poses_list: list[list[dict[str, Any]]] = []
        self.parts_trajectory: dict[str, list[list[float]]] = {}
        self.initial_heights: dict[str, float] = {}
        self.last_positions: dict[str, list[float]] = {}
        self.static_parts: set[str] = set()
        self.released_parts: set[str] = set()

    def reset(self) -> None:
        self.parts_poses_list.clear()
        self.parts_trajectory.clear()
        self.initial_heights.clear()
        self.last_positions.clear()
        self.static_parts.clear()
        self.released_parts.clear()

    def add_parts_poses(self, parts_poses_dict: list[dict[str, Any]], static_window: int = 10, static_threshold: float = 0.005) -> None:
        """记录一步零件位姿，更新轨迹、初始高度和静止判定。

        静止判定：最近 static_window 帧内所有位姿的最大位移 <= static_threshold 时标记为静止。
        """
        self.parts_poses_list.append(parts_poses_dict)
        for part_info in parts_poses_dict:
            prim_path = str(part_info["prim_path"])
            position = [float(v) for v in part_info["position"][:3]]
            if prim_path not in self.initial_heights:
                self.initial_heights[prim_path] = float(position[2])
            self.parts_trajectory.setdefault(prim_path, []).append(position)
            self.last_positions[prim_path] = position

            recent = self.parts_trajectory[prim_path][-int(static_window):]
            if len(recent) >= int(static_window):
                arr = np.asarray(recent, dtype=np.float32)
                max_delta = float(np.max(np.linalg.norm(arr - arr[-1], axis=1)))
                if max_delta <= float(static_threshold):
                    self.static_parts.add(prim_path)

    def update_release_state(
        self,
        parts_poses_dict: list[dict[str, Any]],
        ee_poses: dict[str, Any],
        release_distance_threshold: float = 0.08,
    ) -> list[dict[str, Any]]:
        """更新零件的释放和静止状态，返回注入 released/static 字段的零件列表。

        释放判定：零件到所有端执行器的最近距离 > release_distance_threshold。
        一旦判定为释放则永久记录（released_parts 集合只增不减）。
        """
        ee_points = []
        for value in ee_poses.values():
            vec = np.asarray(value, dtype=np.float32).reshape(-1)
            if vec.size >= 3:
                ee_points.append(vec[:3])

        enriched = []
        for part in parts_poses_dict:
            new_part = dict(part)
            part_id = str(part.get("prim_path", ""))
            pos = np.asarray(part.get("position", []), dtype=np.float32).reshape(-1)
            released = False
            if pos.size >= 3 and ee_points:
                min_distance = min(float(np.linalg.norm(pos[:3] - ee)) for ee in ee_points)
                released = min_distance > float(release_distance_threshold)
            if released:
                self.released_parts.add(part_id)
            new_part["released"] = part_id in self.released_parts
            new_part["static"] = part_id in self.static_parts
            enriched.append(new_part)
        return enriched

    def check_parts_moved(self, movement_threshold: float = 0.001) -> dict[str, bool]:
        moved_result = {}
        for prim_path, trajectory in self.parts_trajectory.items():
            if len(trajectory) < 2:
                moved_result[prim_path] = False
                continue
            start_pos = trajectory[0]
            end_pos = trajectory[-1]
            displacement = np.sqrt(
                (end_pos[0] - start_pos[0])**2 +
                (end_pos[1] - start_pos[1])**2 +
                (end_pos[2] - start_pos[2])**2
            )
            moved_result[prim_path] = displacement > movement_threshold
        return moved_result

    def get_all_parts_moved(self, movement_threshold: float = 0.001) -> bool:
        moved_result = self.check_parts_moved(movement_threshold)
        return all(moved_result.values()) if moved_result else False


def get_robot_ee_poses(robot: WalkerS2sim) -> dict[str, Any]:
    """安全获取机器人端执行器位姿。

    通过 robot._robot_interface.get_ee_poses() 获取各臂末端位姿字典。
    接口不可用时返回空字典。

    Returns:
        dict[str, Any]: 端执行器位姿字典，如 {"left": np.ndarray, "right": np.ndarray}。
    """
    interface = getattr(robot, "_robot_interface", None)
    if interface is None or not hasattr(interface, "get_ee_poses"):
        return {}
    poses = interface.get_ee_poses()
    return poses if isinstance(poses, dict) else {}

class NoOpAssertion(TaskAssertion):
    """不做任何判断，仅靠超时终止 episode。适用于 task3 暂无判据的情况。"""
    def __call__(
        self,
        robot: WalkerS2sim,
        step: int,
        action: np.ndarray | torch.Tensor | None = None,
        extra_info: any = None,
    ) -> tuple[bool, bool, str, dict]:
        terminal = check_step_terminal(step, robot.config.task_name.upper())
        reason = "达到最大步数" if terminal else ""
        return False, terminal, reason, {}

class Task1Assertion(TaskAssertion):
    """Task1（抓取-放置）断言（官方标准）。

    评分规则：
    - 抓取: 每零件 10 分，相对初始高度抬升 >= lift_delta(0.10m) 计分，满分 40。
    - 放置: 每零件 10 分，零件在正确料箱内、已释放且静止后计分，满分 40。
    - 时间: 满分 20，180 秒内完成，每超 30 秒扣 5 分。
    - 成功门槛: 80 分（抓取+放置），达标后才计算时间分。

    终止条件: 零件出界 / 超时。
    """
    def __init__(
        self,
        lift_height: float,
        workspace_limits: dict[str, tuple[float, float]],
        box_half_size: tuple[float, float, float],
        lift_score_per_part: int,
        box_score_per_part: int,
        max_parts: int,
        time_full_score: int,
        time_full_time_seconds: float,
        time_penalty_interval_seconds: float,
        time_penalty_per_interval: int,
        success_score_threshold: int,
        parts_movement_threshold: float,
        lift_delta: float,
    ):
        self._lift_height = float(lift_height)
        self._workspace_limits = workspace_limits
        if len(box_half_size) != 3:
            raise ValueError("task1_box_half_size 必须包含 3 个元素: [x, y, z]")
        self._box_half_size: tuple[float, float, float] = (
            float(box_half_size[0]),
            float(box_half_size[1]),
            float(box_half_size[2]),
        )
        self._lift_score_per_part = int(lift_score_per_part)
        self._box_score_per_part = int(box_score_per_part)
        self._max_parts = int(max_parts)
        self._time_full_score = int(time_full_score)
        self._time_full_time_seconds = float(time_full_time_seconds)
        self._time_penalty_interval_seconds = float(time_penalty_interval_seconds)
        self._time_penalty_per_interval = int(time_penalty_per_interval)
        self._success_score_threshold = int(success_score_threshold)
        self._parts_movement_threshold = float(parts_movement_threshold)
        self._lift_delta = float(lift_delta)
        self._lift_scored_parts: set[str] = set()
        self._box_scored_parts: set[str] = set()
        self._episode_start_time: float = time.time()
        self._last_step: int = 0
        self._parts_tracker = EpisodePartsTracker()

    def _reset_episode_state(self) -> None:
        """重置 episode 内所有累积计分状态、计时和零件跟踪器。"""
        self._lift_scored_parts.clear()
        self._box_scored_parts.clear()
        self._episode_start_time = time.time()
        self._parts_tracker.reset()

    def __call__(
        self,
        robot: WalkerS2sim,
        step: int,
        action: np.ndarray | torch.Tensor | None = None,
        extra_info: any = None

    ) -> tuple[bool, bool, str, dict]:
        metrics: dict = {}
        try:
            if step <= 1 or step < self._last_step:
                self._reset_episode_state()
            self._last_step = step

            parts_poses = robot._scene_builder.get_parts_world_poses()

            ee_poses = get_robot_ee_poses(robot)
            self._parts_tracker.add_parts_poses(parts_poses)
            parts_poses = self._parts_tracker.update_release_state(parts_poses, ee_poses)

            terminal_out_of_workspace = task1_check_parts_out_of_workspace(
                parts_poses,
                self._workspace_limits,
            )
            terminal_time_out = check_step_terminal(step, extra_info.get("max_steps", 1000))
            terminal_no_movement = False
            parts_moved = False
            terminal = terminal_out_of_workspace or terminal_time_out or terminal_no_movement

            box_score = 0
            lift_score = 0
            time_score = 0
            elapsed_seconds = 0.0

            elapsed_seconds = max(0.0, time.time() - self._episode_start_time)
            box_pos, box_ori = robot._scene_builder.boxes.get_world_poses()
            box_score, self._box_scored_parts = task1_check_parts_in_box(
                box_poses=(np.asarray(box_pos).squeeze(), np.asarray(box_ori).squeeze()),
                parts_poses_dict=parts_poses,
                scored_parts=self._box_scored_parts,
                box_half_size=self._box_half_size,
                score_per_part=self._box_score_per_part,
                max_parts=self._max_parts,
                require_released=True,
                require_static=True,
            )
            lift_score, self._lift_scored_parts = check_parts_lifted_from_initial_height(
                parts_poses_dict=parts_poses,
                initial_heights=self._parts_tracker.initial_heights,
                scored_parts=self._lift_scored_parts,
                lift_delta=self._lift_delta,
                score_per_part=self._lift_score_per_part,
                max_parts=self._max_parts,
            )

            total_score = int(box_score + lift_score)

            is_success = total_score >= self._success_score_threshold
            if is_success:
                time_score = task1_time_out_check(
                    elapsed_seconds=elapsed_seconds,
                    full_score=self._time_full_score,
                    full_time_seconds=self._time_full_time_seconds,
                    penalty_interval_seconds=self._time_penalty_interval_seconds,
                    penalty_per_interval=self._time_penalty_per_interval,
                )
                total_score = int(total_score + time_score)

                parts_moved = self._parts_tracker.get_all_parts_moved(self._parts_movement_threshold)
                terminal_no_movement = not parts_moved

            metrics = {
                "task1_lift_score": int(lift_score),
                "task1_box_score": int(box_score),
                "task1_time_score": int(time_score),
                "task1_total_score": int(total_score),
                "task1_elapsed_seconds": float(elapsed_seconds),
                "task1_lift_scored_count": int(len(self._lift_scored_parts)),
                "task1_box_scored_count": int(len(self._box_scored_parts)),
                "task1_lift_height": float(self._lift_height),
                "task1_box_half_size": [float(v) for v in self._box_half_size],
                "task1_max_parts": int(self._max_parts),
                "task1_success_score_threshold": int(self._success_score_threshold),
                "is_success": bool(is_success),
                "terminal": bool(terminal),
                "task1_terminal_out_of_workspace": bool(terminal_out_of_workspace),
                "task1_terminal_time_out": bool(terminal_time_out),
                "task1_parts_moved": bool(parts_moved),
                "task1_terminal_no_movement": bool(terminal_no_movement),
            }

            if is_success:
                reason = f"Success!!!，总分达到{self._success_score_threshold}"
            elif terminal_no_movement:
                reason = "Terminal!!!，零件未发生运动"
            elif terminal_out_of_workspace:
                reason = "Terminal!!!，零件超出工作空间"
            elif terminal_time_out:
                reason = "Terminal!!!，达到最大步数时间限制"
            else:
                reason = "进行中"
            return is_success, terminal, reason, metrics

        except Exception as e:
            logger.warning(f"Task1 断言异常: {e}")
            return False, False, f"断言异常: {e}", metrics

class Task2Assertion(TaskAssertion):
    """Task2（传送带分拣）断言（官方标准）。

    评分规则：
    - 跟随: 每零件 2.5 分，任一端执行器到达 10 cm 内计分，满分 20。
    - 抓取: 每零件 5 分，相对初始高度抬升 >= lift_delta(0.10m) 计分，满分 40。
    - 分拣: 每零件 5 分，零件在正确料箱内、已释放且静止后计分，满分 40。
    - 总计: 跟随 + 抓取 + 分拣，满分 100。
    - 成功门槛: 100 分。

    终止条件: 超时 / 所有零件掉落传送带以下。
    """
    def __init__(
        self,
        conveyor_limits: dict[str, tuple[float, float]],
        conveyor_drop_z: float,
        bin_half_size: tuple[float, float, float],
        grab_score_per_part: float,
        sort_score_per_part: float,
        max_parts: int,
        success_score_threshold: int,
        follow_score_per_part: float,
        follow_distance_threshold: float,
        lift_delta: float,
    ):
        self._conveyor_limits = conveyor_limits
        self._conveyor_drop_z = float(conveyor_drop_z)
        if len(bin_half_size) != 3:
            raise ValueError("task2_bin_half_size 必须包含 3 个元素: [x, y, z]")
        self._bin_half_size: tuple[float, float, float] = (
            float(bin_half_size[0]),
            float(bin_half_size[1]),
            float(bin_half_size[2]),
        )
        self._grab_score_per_part = float(grab_score_per_part)
        self._sort_score_per_part = float(sort_score_per_part)
        self._max_parts = int(max_parts)
        self._success_score_threshold = int(success_score_threshold)
        self._follow_score_per_part = float(follow_score_per_part)
        self._follow_distance_threshold = float(follow_distance_threshold)
        self._lift_delta = float(lift_delta)
        self._grab_scored_parts: set[str] = set()
        self._sort_scored_parts: set[str] = set()
        self._follow_scored_parts: set[str] = set()
        self._last_step: int = 0
        self._parts_tracker = EpisodePartsTracker()

    def _reset_episode_state(self) -> None:
        self._grab_scored_parts.clear()
        self._sort_scored_parts.clear()
        self._follow_scored_parts.clear()
        self._parts_tracker.reset()

    def __call__(
        self,
        robot: WalkerS2sim,
        step: int,
        action: np.ndarray | torch.Tensor | None = None,
        extra_info: any = None
    ) -> tuple[bool, bool, str, dict]:
        metrics: dict = {}
        try:
            if step <= 1 or step < self._last_step:
                self._reset_episode_state()
            self._last_step = step

            parts_poses = robot._scene_builder.get_parts_world_poses()

            ee_poses = get_robot_ee_poses(robot)
            self._parts_tracker.add_parts_poses(parts_poses)
            parts_poses = self._parts_tracker.update_release_state(parts_poses, ee_poses)

            bin_positions, bin_orientations = robot._scene_builder.boxes.get_world_poses()
            bin_positions = np.asarray(bin_positions)
            bin_orientations = np.asarray(bin_orientations)
            n_bins = int(bin_positions.shape[0])
            if n_bins != 2:
                raise ValueError(f"Task2 需要 2 个料箱，当前为 {n_bins} 个")
            li, ri = np.argsort(bin_positions[:, 0])[:2]
            li, ri = int(li), int(ri)
            left_bin = (bin_positions[li], bin_orientations[li])
            right_bin = (bin_positions[ri], bin_orientations[ri])

            follow_score, self._follow_scored_parts, follow_details = task2_check_end_effector_followed_parts(
                parts_poses_dict=parts_poses,
                ee_poses=ee_poses,
                scored_parts=self._follow_scored_parts,
                distance_threshold=self._follow_distance_threshold,
                score_per_part=self._follow_score_per_part,
                max_parts=self._max_parts,
            )

            grab_score, self._grab_scored_parts, grab_details = task2_check_parts_grabbed(
                parts_poses_dict=parts_poses,
                conveyor_limits=self._conveyor_limits,
                scored_parts=self._grab_scored_parts,
                score_per_part=self._grab_score_per_part,
                max_parts=self._max_parts,
                initial_heights=self._parts_tracker.initial_heights,
                lift_delta=self._lift_delta,
            )

            sort_score, self._sort_scored_parts, sort_details = task2_check_parts_in_correct_bin(
                parts_poses_dict=parts_poses,
                left_bin_pose=left_bin,
                right_bin_pose=right_bin,
                bin_half_size=self._bin_half_size,
                scored_parts=self._sort_scored_parts,
                score_per_part=self._sort_score_per_part,
                max_parts=self._max_parts,
                require_released=True,
                require_static=True,
            )

            total_score = float(follow_score + grab_score + sort_score)
            base_score = total_score

            is_success = base_score >= self._success_score_threshold

            max_steps = extra_info.get("max_steps", 1000) if extra_info else 1000
            terminal_time = check_step_terminal(step, int(max_steps))
            terminal_lost = task2_check_all_parts_lost(
                parts_poses_dict=parts_poses,
                conveyor_z_min=self._conveyor_drop_z,
            )
            terminal = terminal_time or terminal_lost

            metrics = {
                "task2_follow_score": float(follow_score),
                "task2_grab_score": int(grab_score),
                "task2_sort_score": int(sort_score),
                "task2_base_score": float(base_score),
                "task2_total_score": float(total_score),
                "task2_follow_scored_count": int(len(self._follow_scored_parts)),
                "task2_grab_scored_count": int(len(self._grab_scored_parts)),
                "task2_sort_scored_count": int(len(self._sort_scored_parts)),
                "task2_follow_details": follow_details,
                "task2_grab_details": grab_details,
                "task2_sort_details": sort_details,
                "task2_max_parts": int(self._max_parts),
                "task2_success_score_threshold": int(self._success_score_threshold),
                "task2_terminal_time": bool(terminal_time),
                "task2_terminal_lost": bool(terminal_lost),
                "is_success": bool(is_success),
                "terminal": bool(terminal),
            }

            if is_success:
                reason = f"Success！！！，基础分{base_score}达到门槛{self._success_score_threshold}"
            elif terminal_lost:
                reason = "Terminal！！！，所有零件掉落传送带以下"
            elif terminal_time:
                reason = "Terminal！！！，达到最大步数时间限制"
            else:
                reason = "进行中"
            return is_success, terminal, reason, metrics

        except Exception as e:
            logger.warning(f"Task2 断言异常: {e}")
            return False, False, f"断言异常: {e}", metrics


class Task3Assertion(TaskAssertion):
    """Task3（嵌装）断言（官方标准）。

    评分规则：
    - 抓取: 每零件 7.5 分，相对初始高度抬升 >= lift_delta(0.10m) 计分，满分 45。
    - 嵌装: 每零件 7.5 分，匹配槽位内释放静止后计分，每槽位限用一次，满分 45。
    - 时间: 满分 10，360 秒内完成，每超 60 秒扣 5 分。
    - 成功门槛: 90 分（抓取+嵌装），达标后才计算时间分。

    终止条件: 超时 / 零件出界 / 成功。
    前 60 步为环境初始化宽限期，不做出界判死。
    """

    def __init__(
        self,
        foam_pos: list[float],
        workspace_limits: dict,
        dist_threshold: float,
        height_threshold: float,
        success_score_threshold: int,
        grab_score_per_part: float,
        insert_score_per_part: float,
        max_parts: int,
        lift_delta: float,
        time_full_score: int,
        time_full_time_seconds: float,
        time_penalty_interval_seconds: float,
        time_penalty_per_interval: int,
    ):
        self.workspace_limits = workspace_limits
        self.dist_threshold = dist_threshold
        self.height_threshold = height_threshold
        self.success_score_threshold = success_score_threshold
        self.grab_score_per_part = float(grab_score_per_part)
        self.insert_score_per_part = float(insert_score_per_part)
        self.max_parts = int(max_parts)
        self.lift_delta = float(lift_delta)
        self.time_full_score = int(time_full_score)
        self.time_full_time_seconds = float(time_full_time_seconds)
        self.time_penalty_interval_seconds = float(time_penalty_interval_seconds)
        self.time_penalty_per_interval = int(time_penalty_per_interval)

        self.foam_center = np.array(foam_pos)
        self._episode_start_time = time.time()
        self.max_insertion_score = 0
        self._last_step = 0
        self._grab_scored_parts: set[str] = set()
        self._insert_scored_parts: set[str] = set()
        self._used_slots: set[str] = set()
        self._parts_tracker = EpisodePartsTracker()

    def _build_slots(self) -> list[dict]:
        """构建 Task3 嵌装槽位列表（A 类 Y=0.21，B 类 Y=0.41，各 3 个）。

        Returns:
            list[dict]: 槽位信息，每项含 slot_id、type、position。
        """
        return [
            {"slot_id": "A0", "type": "A", "position": [0.54, 0.21, 1.04]},
            {"slot_id": "A1", "type": "A", "position": [0.76, 0.21, 1.04]},
            {"slot_id": "A2", "type": "A", "position": [0.98, 0.21, 1.04]},
            {"slot_id": "B0", "type": "B", "position": [0.54, 0.41, 1.04]},
            {"slot_id": "B1", "type": "B", "position": [0.76, 0.41, 1.04]},
            {"slot_id": "B2", "type": "B", "position": [0.98, 0.41, 1.04]},
        ]

    def _reset_episode_state(self) -> None:
        """重置 episode 内所有累积状态（计分、槽位、零件跟踪器、计时）。"""
        self._episode_start_time = time.time()
        self.max_insertion_score = 0
        self._grab_scored_parts.clear()
        self._insert_scored_parts.clear()
        self._used_slots.clear()
        self._parts_tracker.reset()

    def __call__(self, robot, step, action=None, extra_info=None, **kwargs):
        if step <= 1 or step < self._last_step:
            self._reset_episode_state()
        self._last_step = step

        elapsed = time.time() - self._episode_start_time

        is_grace_period = (step < 60)
        parts_poses = robot._scene_builder.get_parts_world_poses()
        terminal_time = check_step_terminal(step, extra_info.get("max_steps", 10000) if extra_info else 10000)

        ee_poses = get_robot_ee_poses(robot)
        self._parts_tracker.add_parts_poses(parts_poses)
        parts_poses = self._parts_tracker.update_release_state(parts_poses, ee_poses)

        terminal_out = False
        reason_out = ""
        if not is_grace_period:
            terminal_out, reason_out = task3_check_terminal(
                robot, parts_poses, self.foam_center, self.workspace_limits
            )

        grab_score, self._grab_scored_parts = check_parts_lifted_from_initial_height(
            parts_poses_dict=parts_poses,
            initial_heights=self._parts_tracker.initial_heights,
            scored_parts=self._grab_scored_parts,
            lift_delta=self.lift_delta,
            score_per_part=self.grab_score_per_part,
            max_parts=self.max_parts,
        )

        insert_score, self._insert_scored_parts, self._used_slots = task3_check_parts_inserted_in_slots(
            parts_poses_dict=parts_poses,
            slots=self._build_slots(),
            scored_parts=self._insert_scored_parts,
            used_slots=self._used_slots,
            dist_threshold=self.dist_threshold,
            height_threshold=self.height_threshold,
            score_per_part=self.insert_score_per_part,
            max_parts=self.max_parts,
            require_released=True,
            require_static=True,
        )

        base_score = float(grab_score + insert_score)
        is_success = base_score >= float(self.success_score_threshold)

        time_score = 0
        if is_success:
            time_score = int(calculate_time_score(
                elapsed_seconds=elapsed,
                full_score=self.time_full_score,
                full_time_seconds=self.time_full_time_seconds,
                penalty_interval_seconds=self.time_penalty_interval_seconds,
                penalty_per_interval=self.time_penalty_per_interval,
            ))
        total_score = float(base_score + time_score)

        terminal = terminal_time or terminal_out or is_success

        reason = "进行中"
        if is_success: reason = "成功完成"
        elif terminal_time: reason = "超时"
        elif terminal_out: reason = reason_out or "出界(零件掉落)"
        elif is_grace_period: reason = "环境初始化中"

        metrics = {
            "task3_grab_score": float(grab_score),
            "task3_insert_score": float(insert_score),
            "task3_time_score": int(time_score),
            "task3_total_score": float(total_score),
            "task3_grab_scored_count": int(len(self._grab_scored_parts)),
            "task3_insert_scored_count": int(len(self._insert_scored_parts)),
            "task3_used_slot_count": int(len(self._used_slots)),
            "elapsed_time": float(elapsed),
            "total_score": float(total_score),
        }

        return is_success, terminal, reason, metrics

class Task4Assertion(TaskAssertion):
    """Task4（装箱）断言（官方标准）。

    评分规则：
    - 短边: 每边 15 分，需端执行器接触对应边缘位置后闭合，满分 30。
    - 长边: 每边 15 分，需端执行器接触对应边缘位置后闭合，满分 30。
    - 时间: 满分 10，180 秒内完成，每超 30 秒扣 5 分。
    - 无协作系数（官方标准中不存在）。
    - 总分上限 100。

    成功条件: 连续 success_hold_steps 步盒子关节达到目标角度。
    终止条件: 盒子位姿偏离初始位姿 / 超时 / 成功。
    """

    def __init__(
        self,
        short_edge_targets: tuple[float, float],
        long_edge_targets: tuple[float, float],
        short_edge_joint_indices: tuple[int, int],
        long_edge_joint_indices: tuple[int, int],
        joint_threshold: float,
        action_movement_eps: float,
        co_move_ratio_threshold: float,
        success_hold_steps: int,
        short_edge_score_per_edge: int,
        long_edge_score_per_edge: int,
        time_full_score: int,
        time_full_time_seconds: float,
        time_penalty_interval_seconds: float,
        time_penalty_per_interval: int,
        box_pose_position_threshold: float,
        box_pose_orientation_threshold: float,
        contact_distance_threshold: float,
    ):
        self._short_edge_targets = np.asarray(short_edge_targets, dtype=np.float32).reshape(2)
        self._long_edge_targets = np.asarray(long_edge_targets, dtype=np.float32).reshape(2)
        self._short_edge_joint_indices = short_edge_joint_indices
        self._long_edge_joint_indices = long_edge_joint_indices
        self._joint_threshold = float(joint_threshold)
        self._action_movement_eps = float(action_movement_eps)
        self._co_move_ratio_threshold = float(co_move_ratio_threshold)
        self._success_hold = success_hold_steps
        self._short_edge_score_per_edge = short_edge_score_per_edge
        self._long_edge_score_per_edge = long_edge_score_per_edge
        self._time_full_score = time_full_score
        self._time_full_time_seconds = time_full_time_seconds
        self._time_penalty_interval_seconds = time_penalty_interval_seconds
        self._time_penalty_per_interval = time_penalty_per_interval
        self._box_pose_position_threshold = box_pose_position_threshold
        self._box_pose_orientation_threshold = box_pose_orientation_threshold
        self._contact_distance_threshold = float(contact_distance_threshold)
        self._consecutive_success = 0
        self._episode_start_time = time.time()
        self._last_step: int = 0
        self._action_tracker = Task4EpisodeActionTracker()
        self._contacted_short_edges: set[int] = set()
        self._contacted_long_edges: set[int] = set()

    @property
    def target_box_joints(self) -> np.ndarray:
        return np.asarray([
            [
                float(self._long_edge_targets[0]),
                float(self._long_edge_targets[1]),
                float(self._short_edge_targets[0]),
                float(self._short_edge_targets[1]),
            ]
        ], dtype=np.float32)

    def _reset_episode_state(self) -> None:
        self._episode_start_time = time.time()
        self._consecutive_success = 0
        self._action_tracker.reset()
        self._contacted_short_edges.clear()
        self._contacted_long_edges.clear()

    def _update_edge_contacts(self, robot: WalkerS2sim) -> None:
        """基于端执行器与盒子边缘的接近距离更新接触状态。

        接触判定：任一端执行器到边缘参考点的距离 <= contact_distance_threshold。
        一旦接触则永久记录（_contacted_short_edges/_contacted_long_edges 只增不减）。
        若 Isaac Sim 原生接触传感器可用，可替换本方法内部实现。
        """
        ee_poses = get_robot_ee_poses(robot)
        ee_points = []
        for value in ee_poses.values():
            vec = np.asarray(value, dtype=np.float32).reshape(-1)
            if vec.size >= 3:
                ee_points.append(vec[:3])
        if not ee_points:
            return

        box_pos, _ = robot._scene_builder.box_articulation.get_world_poses()
        center = np.asarray(box_pos, dtype=np.float32).reshape(-1)[:3]
        short_edge_points = [
            center + np.array([0.0, -0.20, 0.12], dtype=np.float32),
            center + np.array([0.0, 0.20, 0.12], dtype=np.float32),
        ]
        long_edge_points = [
            center + np.array([-0.30, 0.0, 0.12], dtype=np.float32),
            center + np.array([0.30, 0.0, 0.12], dtype=np.float32),
        ]

        for idx, edge_point in enumerate(short_edge_points):
            if min(float(np.linalg.norm(edge_point - ee)) for ee in ee_points) <= self._contact_distance_threshold:
                self._contacted_short_edges.add(idx)
        for idx, edge_point in enumerate(long_edge_points):
            if min(float(np.linalg.norm(edge_point - ee)) for ee in ee_points) <= self._contact_distance_threshold:
                self._contacted_long_edges.add(idx)

    def __call__(
        self,
        robot: WalkerS2sim,
        step: int,
        action: np.ndarray | torch.Tensor | None = None,
        extra_info: any = None,
    ) -> tuple[bool, bool, str, dict]:
        if step <= 1 or step < self._last_step:
            self._reset_episode_state()
        self._last_step = step
        self._action_tracker.add_action(action)

        # 终止条件1：盒子位姿偏离
        cur_pos, cur_ori = robot._scene_builder.box_articulation.get_world_poses()
        init_pos = robot._scene_builder._box_initial_world_pos
        init_ori = robot._scene_builder._box_initial_world_ori
        current_pose = np.concatenate([cur_pos.squeeze(), cur_ori.squeeze()])
        target_pose = np.concatenate([init_pos.squeeze(), init_ori.squeeze()])
        terminal_pose = task4_check_box_poses_terminal(
            current_pose,
            target_pose,
            position_threshold=self._box_pose_position_threshold,
            orientation_threshold=self._box_pose_orientation_threshold,
        )

        # 终止条件2：超时
        terminal_time = check_step_terminal(step, extra_info.get("max_steps", 1000))

        terminal = terminal_pose or terminal_time

        # 成功条件：盒子关节角达到目标
        cur_joints = robot.get_box_joints()
        is_raw_success, error_info = task4_check_box_joints_success(
            cur_joints,
            self.target_box_joints,
            threshold=self._joint_threshold,
        )

        if is_raw_success:
            self._consecutive_success += 1
        else:
            self._consecutive_success = 0

        # 连续成功达到要求才算真正成功，避免机器人一碰最后一个边就退出
        is_success = self._consecutive_success >= self._success_hold
        if is_success:
            terminal = True

        self._update_edge_contacts(robot)

        short_edge_raw_score, short_info = task4_check_short_edge_close_score(
            current_box_joints=cur_joints,
            short_edge_targets=self._short_edge_targets,
            joint_indices=self._short_edge_joint_indices,
            threshold=self._joint_threshold,
            score_per_edge=self._short_edge_score_per_edge,
        )
        long_edge_raw_score, long_info = task4_check_long_edge_close_score(
            current_box_joints=cur_joints,
            long_edge_targets=self._long_edge_targets,
            joint_indices=self._long_edge_joint_indices,
            threshold=self._joint_threshold,
            score_per_edge=self._long_edge_score_per_edge,
        )

        short_edge_score = sum(
            self._short_edge_score_per_edge
            for idx, closed in enumerate(short_info.get("short_edge_closed", []))
            if closed and idx in self._contacted_short_edges
        )
        long_edge_score = sum(
            self._long_edge_score_per_edge
            for idx, closed in enumerate(long_info.get("long_edge_closed", []))
            if closed and idx in self._contacted_long_edges
        )

        elapsed_seconds = max(0.0, time.time() - self._episode_start_time)
        time_score = task4_time_score(
            elapsed_seconds=elapsed_seconds,
            full_score=self._time_full_score,
            full_time_seconds=self._time_full_time_seconds,
            penalty_interval_seconds=self._time_penalty_interval_seconds,
            penalty_per_interval=self._time_penalty_per_interval,
        )

        if not is_success:
            time_score = 0

        score_info = task4_calculate_total_score(
            short_edge_score=short_edge_score,
            long_edge_score=long_edge_score,
            time_score=time_score,
        )
        total_score = int(score_info.get("final_score", 0))
        task4_time_score_val = int(time_score)

        metrics = {
            "current_box_joints": cur_joints.tolist() if hasattr(cur_joints, "tolist") else cur_joints,
            "target_box_joints": self.target_box_joints.tolist(),
            "max_error": error_info.get("max_error"),
            "mean_error": error_info.get("mean_error"),
            "all_errors": error_info.get("all_errors"),
            "task4_short_edge_score": int(short_edge_score),
            "task4_long_edge_score": int(long_edge_score),
            "task4_time_score": task4_time_score_val,
            "task4_raw_score": int(score_info.get("raw_score", 0)),
            "task4_total_score": int(total_score),
            "task4_short_edge_closed": short_info.get("short_edge_closed", [False, False]),
            "task4_short_edge_errors": short_info.get("short_edge_errors", [float("inf"), float("inf")]),
            "task4_long_edge_closed": long_info.get("long_edge_closed", [False, False]),
            "task4_long_edge_errors": long_info.get("long_edge_errors", [float("inf"), float("inf")]),
            "task4_elapsed_seconds": float(elapsed_seconds),
            "task4_contacted_short_edges": int(len(self._contacted_short_edges)),
            "task4_contacted_long_edges": int(len(self._contacted_long_edges)),
            "is_success": is_success,
            "terminal_pose": terminal_pose,
            "terminal_time": terminal_time,
            "terminal": terminal,
            "total_score": int(total_score),
        }

        reason = (
            "盒子关节达到目标，装箱成功" if is_success
            else ("盒子位姿偏离" if terminal_pose else ("超时" if terminal_time else "进行中"))
        )
        return is_success, terminal, reason, metrics


# ─────────────────────────────────────────────
# Task Assertion Registry
# ─────────────────────────────────────────────
TASK_ASSERTION_REGISTRY: dict[str, type[TaskAssertion]] = {
    "task1": Task1Assertion,
    "task2": Task2Assertion,
    "task3": Task3Assertion,
    "task4": Task4Assertion,
}


def create_task_assertion(task: str, args) -> TaskAssertion:
    """根据任务名和配置参数创建对应的断言实例。"""
    task = task.lower()

    if task == "task1":
        workspace_limits = getattr(args, "task1_workspace_limits")
        if not isinstance(workspace_limits, dict):
            raise ValueError("task1_workspace_limits 必须是包含 x/y/z 键的字典")
        return Task1Assertion(
            lift_height=float(getattr(args, "task1_lift_height")),
            workspace_limits={
                "x": (float(workspace_limits["x"][0]), float(workspace_limits["x"][1])),
                "y": (float(workspace_limits["y"][0]), float(workspace_limits["y"][1])),
                "z": (float(workspace_limits["z"][0]), float(workspace_limits["z"][1])),
            },
            box_half_size=tuple(getattr(args, "task1_box_half_size")),
            lift_score_per_part=int(getattr(args, "task1_lift_score_per_part")),
            box_score_per_part=int(getattr(args, "task1_box_score_per_part")),
            max_parts=int(getattr(args, "task1_max_parts")),
            time_full_score=int(getattr(args, "task1_time_full_score")),
            time_full_time_seconds=float(getattr(args, "task1_time_full_time_seconds")),
            time_penalty_interval_seconds=float(getattr(args, "task1_time_penalty_interval_seconds")),
            time_penalty_per_interval=int(getattr(args, "task1_time_penalty_per_interval")),
            success_score_threshold=int(getattr(args, "task1_success_score_threshold")),
            parts_movement_threshold=float(getattr(args, "task1_parts_movement_threshold")),
            lift_delta=float(getattr(args, "task1_lift_delta")),
        )

    elif task == "task4":
        return Task4Assertion(
            short_edge_targets=tuple(getattr(args, "task4_short_targets")),
            long_edge_targets=tuple(getattr(args, "task4_long_targets")),
            short_edge_joint_indices=tuple(getattr(args, "task4_short_edge_joint_indices")),
            long_edge_joint_indices=tuple(getattr(args, "task4_long_edge_joint_indices")),
            joint_threshold=float(getattr(args, "task4_joint_threshold")),
            action_movement_eps=float(getattr(args, "task4_action_movement_eps")),
            co_move_ratio_threshold=float(getattr(args, "task4_co_move_ratio_threshold")),
            success_hold_steps=int(getattr(args, "task4_success_hold_steps")),
            short_edge_score_per_edge=int(getattr(args, "task4_short_edge_score_per_edge")),
            long_edge_score_per_edge=int(getattr(args, "task4_long_edge_score_per_edge")),
            time_full_score=int(getattr(args, "task4_time_full_score")),
            time_full_time_seconds=float(getattr(args, "task4_time_full_time_seconds")),
            time_penalty_interval_seconds=float(getattr(args, "task4_time_penalty_interval_seconds")),
            time_penalty_per_interval=int(getattr(args, "task4_time_penalty_per_interval")),
            box_pose_position_threshold=float(getattr(args, "task4_box_pose_position_threshold")),
            box_pose_orientation_threshold=float(getattr(args, "task4_box_pose_orientation_threshold")),
            contact_distance_threshold=float(getattr(args, "task4_contact_distance_threshold")),
        )

    elif task == "task2":
        conveyor_limits = getattr(args, "task2_conveyor_limits")
        if not isinstance(conveyor_limits, dict):
            raise ValueError("task2_conveyor_limits 必须是包含 x/y/z 键的字典")
        for axis in ["x", "y", "z"]:
            if axis in conveyor_limits:
                conveyor_limits[axis] = [float(conveyor_limits[axis][0]), float(conveyor_limits[axis][1])]
        return Task2Assertion(
            conveyor_limits={
                "x": (float(conveyor_limits["x"][0]), float(conveyor_limits["x"][1])),
                "y": (float(conveyor_limits["y"][0]), float(conveyor_limits["y"][1])),
                "z": (float(conveyor_limits["z"][0]), float(conveyor_limits["z"][1])),
            },
            conveyor_drop_z=float(getattr(args, "task2_conveyor_drop_z")),
            bin_half_size=tuple(getattr(args, "task2_bin_half_size")),
            grab_score_per_part=float(getattr(args, "task2_grab_score_per_part")),
            sort_score_per_part=float(getattr(args, "task2_sort_score_per_part")),
            max_parts=int(getattr(args, "task2_max_parts")),
            success_score_threshold=int(getattr(args, "task2_success_score_threshold")),
            follow_score_per_part=float(getattr(args, "task2_follow_score_per_part")),
            follow_distance_threshold=float(getattr(args, "task2_follow_distance_threshold")),
            lift_delta=float(getattr(args, "task2_lift_delta")),
        )

    elif task == "task3":
        return Task3Assertion(
            foam_pos=list(getattr(args, "task3_foam_pos")),
            workspace_limits=getattr(args, "task3_workspace_limits"),
            dist_threshold=float(getattr(args, "task3_dist_threshold")),
            height_threshold=float(getattr(args, "task3_height_threshold")),
            success_score_threshold=int(getattr(args, "task3_success_score_threshold")),
            grab_score_per_part=float(getattr(args, "task3_grab_score_per_part")),
            insert_score_per_part=float(getattr(args, "task3_insert_score_per_part")),
            max_parts=int(getattr(args, "task3_max_parts")),
            lift_delta=float(getattr(args, "task3_lift_delta")),
            time_full_score=int(getattr(args, "task3_time_full_score")),
            time_full_time_seconds=float(getattr(args, "task3_time_full_time_seconds")),
            time_penalty_interval_seconds=float(getattr(args, "task3_time_penalty_interval_seconds")),
            time_penalty_per_interval=int(getattr(args, "task3_time_penalty_per_interval")),
        )

    else:
        raise ValueError(f"暂不支持的任务类型：{task}")
