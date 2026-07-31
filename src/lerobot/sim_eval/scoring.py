import numpy as np
from typing import Any, Iterable

from .common import PartPoseDict, iter_part_dicts, safe_vec3


def _extract_semantics(part: PartPoseDict) -> str:
    """兼容语义来源：优先 semantics，其次 prim_path。"""
    sem = str(part.get("semantics", "")).strip().upper()
    if sem in {"A", "B"}:
        return sem

    prim_path = str(part.get("prim_path", "")).upper()
    if "PART_A" in prim_path or prim_path.endswith("A"):
        return "A"
    if "PART_B" in prim_path or prim_path.endswith("B"):
        return "B"
    return "UNKNOWN"



def _part_id(part: PartPoseDict, idx: int) -> str:
    """从零件字典中提取唯一标识符，优先使用 prim_path，回退到索引序号。"""
    return str(part.get("prim_path", f"part_{idx}"))


def _is_released(part: PartPoseDict, require_released: bool) -> bool:
    """检查零件是否已释放：若不需要释放条件则始终返回 True，否则读取 released 字段。"""
    if not require_released:
        return True
    return bool(part.get("released", False))


def _is_static(part: PartPoseDict, require_static: bool) -> bool:
    """检查零件是否已静止：若不需要静止条件则始终返回 True，否则读取 static 字段。"""
    if not require_static:
        return True
    return bool(part.get("static", False))


def calculate_time_score(
    elapsed_seconds: float,
    full_score: float,
    full_time_seconds: float,
    penalty_interval_seconds: float,
    penalty_per_interval: float,
) -> float:
    """通用时间评分函数（官方标准）。

    在 full_time_seconds 秒内完成得满分 full_score；
    每超过 penalty_interval_seconds 秒扣 penalty_per_interval 分，最低 0 分。

    Args:
        elapsed_seconds: 实际耗时（秒）。
        full_score: 时间满分。
        full_time_seconds: 获得满分的时间上限（秒）。
        penalty_interval_seconds: 扣分时间间隔（秒）。
        penalty_per_interval: 每个间隔扣除的分数。

    Returns:
        float: 时间得分。
    """
    t = max(0.0, float(elapsed_seconds))
    full = float(full_score)
    if t <= float(full_time_seconds):
        return full
    overtime = t - float(full_time_seconds)
    penalty_steps = int(np.ceil(overtime / float(penalty_interval_seconds)))
    penalty = penalty_steps * float(penalty_per_interval)
    return max(0.0, full - penalty)


def check_parts_lifted_from_initial_height(
    parts_poses_dict: Any,
    initial_heights: dict[str, float],
    scored_parts: Iterable[str] | None = None,
    lift_delta: float = 0.10,
    score_per_part: float = 10.0,
    max_parts: int = 4,
) -> tuple[float, set[str]]:
    """基于初始高度判定零件抬升得分（官方标准）。

    零件当前 z 高度 >= 该零件初始 z 高度 + lift_delta 时计分一次。

    Args:
        parts_poses_dict: 零件位姿列表。
        initial_heights: 每个零件的初始 z 高度映射 {prim_path: z}。
        scored_parts: 已计分的零件 ID 集合。
        lift_delta: 抬升判定的高度差阈值（米），默认 0.10。
        score_per_part: 每个零件得分。
        max_parts: 最大计分零件数。

    Returns:
        (score, scored_parts): 总分和已计分零件集合。
    """
    current_scored_parts: set[str] = set(scored_parts or [])
    parts = iter_part_dicts(parts_poses_dict)
    for idx, part in enumerate(parts):
        part_id = _part_id(part, idx)
        if part_id in current_scored_parts:
            continue
        pos = safe_vec3(part.get("position"))
        if pos is None or part_id not in initial_heights:
            continue
        if float(pos[2]) >= float(initial_heights[part_id]) + float(lift_delta):
            current_scored_parts.add(part_id)
            if len(current_scored_parts) >= int(max_parts):
                break
    score = min(float(score_per_part) * int(max_parts), len(current_scored_parts) * float(score_per_part))
    return score, current_scored_parts


def point_in_aabb(point: np.ndarray, center: np.ndarray, half_size: tuple[float, float, float]) -> bool:
    """判断三维点是否在轴对齐包围盒（AABB）内。

    Args:
        point: 待检测点 (3,)。
        center: 包围盒中心 (3,)。
        half_size: 包围盒半尺寸 (hx, hy, hz)。

    Returns:
        bool: 点在包围盒内返回 True。
    """
    rel = point - center
    hx, hy, hz = float(half_size[0]), float(half_size[1]), float(half_size[2])
    return (
        abs(float(rel[0])) <= hx
        and abs(float(rel[1])) <= hy
        and abs(float(rel[2])) <= hz
    )


# ----------------- Task1 评估工具函数 -----------------

def task1_check_parts_in_box(
    box_poses: tuple[np.ndarray, np.ndarray],
    parts_poses_dict: Any,
    scored_parts: Iterable[str] | None = None,
    box_half_size: tuple[float, float, float] = (0.095, 0.145, 0.25),
    score_per_part: int = 10,
    max_parts: int = 4,
    require_released: bool = False,
    require_static: bool = False,
) -> tuple[int, set[str]]:
    """
    Task1 入箱评分（40分满分，每个工件10分）：
    1. 零件在箱体空间内；
    2. 按 semantics(A/B) 落在正确类别区域；
    3. 一个工件只计分一次。
    """
    current_scored_parts: set[str] = set(scored_parts or [])
    try:
        box_pos = safe_vec3(box_poses[0] if isinstance(box_poses, tuple) and len(box_poses) > 0 else None)
        if box_pos is None:
            raise ValueError(f"task1_check_parts_in_box 异常: 无效的 box_poses 输入，无法获取箱体位置")

        parts = iter_part_dicts(parts_poses_dict)
        hx, hy, hz = float(box_half_size[0]), float(box_half_size[1]), float(box_half_size[2])

        for idx, part in enumerate(parts):
            part_id = str(part.get("prim_path", f"part_{idx}"))
            if part_id in current_scored_parts:
                continue

            pos = safe_vec3(part.get("position"))
            if pos is None:
                continue

            rel = pos - box_pos
            in_box = (abs(float(rel[0])) <= hx) and (abs(float(rel[1])) <= hy) and (abs(float(rel[2])) <= hz)
            if not in_box:
                continue

            sem = _extract_semantics(part)
            # 将箱体按 y 轴分成 A/B 两个类别区域：A 在负半轴，B 在正半轴。
            correct_category = (sem == "B" and rel[1] >= 0.0) or (sem == "A" and rel[1] < 0.0)
            if not correct_category:
                continue

            if not _is_released(part, require_released):
                continue
            if not _is_static(part, require_static):
                continue

            current_scored_parts.add(part_id)
            if len(current_scored_parts) >= int(max_parts):
                break

        score = min(int(score_per_part) * int(max_parts), len(current_scored_parts) * int(score_per_part))
        return score, current_scored_parts
    except Exception as e:
        raise Exception(f"task1_check_parts_in_box 异常: {e}")


def task1_check_parts_in_lift(
    parts_poses_dict: Any,
    threshold_height: float,
    scored_parts: Iterable[str] | None = None,
    score_per_part: int = 10,
    max_parts: int = 4,
) -> tuple[int, set[str]]:
    """
    Task1 抬升评分（40分满分，每个工件10分）。
    判定条件：零件 z 高度 >= 指定阈值；一个工件只计分一次。
    """
    current_scored_parts: set[str] = set(scored_parts or [])
    try:
        parts = iter_part_dicts(parts_poses_dict)
        threshold = float(threshold_height)

        for idx, part in enumerate(parts):
            part_id = str(part.get("prim_path", f"part_{idx}"))
            if part_id in current_scored_parts:
                continue

            pos = safe_vec3(part.get("position"))
            if pos is None:
                continue

            if float(pos[2]) >= threshold:
                current_scored_parts.add(part_id)
                if len(current_scored_parts) >= int(max_parts):
                    break

        score = min(int(score_per_part) * int(max_parts), len(current_scored_parts) * int(score_per_part))
        return score, current_scored_parts
    except Exception as e:
        raise Exception(f"task1_check_parts_in_lift 异常: {e}") 


def task1_time_out_check(
    elapsed_seconds: float,
    full_score: int = 20,
    full_time_seconds: float = 180.0,
    penalty_interval_seconds: float = 30.0,
    penalty_per_interval: int = 5,
) -> int:
    """Task1 时间评分（官方标准，满分 20）。

    180 秒内完成得满分，每超 30 秒扣 5 分，最低 0 分。
    """
    return int(calculate_time_score(
        elapsed_seconds=elapsed_seconds,
        full_score=full_score,
        full_time_seconds=full_time_seconds,
        penalty_interval_seconds=penalty_interval_seconds,
        penalty_per_interval=penalty_per_interval,
    ))



# ----------------- Task2 评估工具函数 -----------------


def task2_check_parts_grabbed(
    parts_poses_dict: Any,
    conveyor_limits: dict[str, tuple[float, float]],
    scored_parts: Iterable[str] | None = None,
    score_per_part: float = 5.0,
    max_parts: int = 8,
    initial_heights: dict[str, float] | None = None,
    lift_delta: float = 0.10,
) -> tuple[int, set[str], dict]:
    """
    Task2 抓取评分：检查工件是否脱离传送带。

    判定条件：
    - 工件当前 z 高度 > 传送带高度 + 阈值
    - 一个工件只计分一次

    Returns:
        score: 抓取总分
        scored_parts: 已计分的工件ID集合
        details: 包含每个工件抓取状态的详细信息
    """
    current_scored_parts: set[str] = set(scored_parts or [])
    details: dict[str, dict] = {}
    try:
        parts = iter_part_dicts(parts_poses_dict)
        z_min = float(conveyor_limits.get("z", (0, 0))[0])
        # 传送带高度 + 一定阈值（如 0.05m）视为脱离传送带
        grab_threshold = z_min + 0.05

        for idx, part in enumerate(parts):
            part_id = str(part.get("prim_path", f"part_{idx}"))
            if part_id in current_scored_parts:
                # 已计分的工件，更新状态
                pos = safe_vec3(part.get("position"))
                if pos is not None:
                    details[part_id] = {
                        "grabbed": True,
                        "z_height": float(pos[2]),
                        "grabbed_at_step": details.get(part_id, {}).get("grabbed_at_step", "already_scored"),
                    }
                continue

            pos = safe_vec3(part.get("position"))
            if pos is None:
                continue

            if initial_heights is not None and part_id in initial_heights:
                is_grabbed = float(pos[2]) >= float(initial_heights[part_id]) + float(lift_delta)
                grab_threshold_val = float(initial_heights[part_id]) + float(lift_delta)
            else:
                is_grabbed = float(pos[2]) > grab_threshold
                grab_threshold_val = float(grab_threshold)
            details[part_id] = {
                "grabbed": bool(is_grabbed),
                "z_height": float(pos[2]),
                "grab_threshold": grab_threshold_val,
            }

            if is_grabbed:
                current_scored_parts.add(part_id)
                details[part_id]["grabbed_at_step"] = "scored"
                if len(current_scored_parts) >= int(max_parts):
                    break

        score = min(float(score_per_part) * int(max_parts), len(current_scored_parts) * float(score_per_part))
        return int(score), current_scored_parts, details
    except Exception as e:
        raise Exception(f"task2_check_parts_grabbed 异常: {e}")


def task2_check_parts_in_correct_bin(
    parts_poses_dict: Any,
    left_bin_pose: tuple[np.ndarray, np.ndarray],
    right_bin_pose: tuple[np.ndarray, np.ndarray],
    bin_half_size: tuple[float, float, float] = (0.15, 0.10, 0.05),
    scored_parts: Iterable[str] | None = None,
    score_per_part: float = 5.0,
    max_parts: int = 8,
    require_released: bool = False,
    require_static: bool = False,
) -> tuple[int, set[str], dict]:
    """
    Task2 分拣评分：检查工件是否在正确类别的料箱内。

    判定条件：
    - 工件在料箱空间内
    - A工件应在左侧料箱，B工件应在右侧料箱
    - 一个工件只计分一次

    Args:
        parts_poses_dict: 工件位姿字典
        left_bin_pose: 左料箱位姿
        right_bin_pose: 右料箱位姿
        bin_half_size: 料箱半尺寸
        scored_parts: 已计分的工件ID集合
        score_per_part: 每个工件得分
        max_parts: 最大工件数

    Returns:
        score: 分拣总分
        scored_parts: 已计分的工件ID集合
        details: 包含每个工件分拣状态的详细信息
    """
    current_scored_parts: set[str] = set(scored_parts or [])
    details: dict[str, dict] = {}
    try:
        left_bin_pos = safe_vec3(left_bin_pose[0] if isinstance(left_bin_pose, tuple) and len(left_bin_pose) > 0 else None)
        right_bin_pos = safe_vec3(right_bin_pose[0] if isinstance(right_bin_pose, tuple) and len(right_bin_pose) > 0 else None)

        if left_bin_pos is None or right_bin_pos is None:
            raise ValueError("task2_check_parts_in_correct_bin 异常: 无效的料箱位置")

        parts = iter_part_dicts(parts_poses_dict)
        hx, hy, hz = float(bin_half_size[0]), float(bin_half_size[1]), float(bin_half_size[2])

        for idx, part in enumerate(parts):
            part_id = str(part.get("prim_path", f"part_{idx}"))
            if part_id in current_scored_parts:
                # 已计分的工件，更新状态
                pos = safe_vec3(part.get("position"))
                if pos is not None:
                    details[part_id] = {
                        "sorted": True,
                        "bin_side": details.get(part_id, {}).get("bin_side", "unknown"),
                        "is_correct": True,
                    }
                continue

            pos = safe_vec3(part.get("position"))
            if pos is None:
                continue

            sem = _extract_semantics(part)
            if sem == "UNKNOWN":
                continue

            # 检查是否在左料箱内
            rel_left = pos - left_bin_pos
            in_left_bin = (abs(float(rel_left[0])) <= hx) and (abs(float(rel_left[1])) <= hy) and (abs(float(rel_left[2])) <= hz)

            # 检查是否在右料箱内
            rel_right = pos - right_bin_pos
            in_right_bin = (abs(float(rel_right[0])) <= hx) and (abs(float(rel_right[1])) <= hy) and (abs(float(rel_right[2])) <= hz)

            # A工件应在左料箱，B工件应在右料箱
            is_correct = (sem == "A" and in_left_bin) or (sem == "B" and in_right_bin)
            if is_correct and (not _is_released(part, require_released) or not _is_static(part, require_static)):
                is_correct = False
            in_any_bin = in_left_bin or in_right_bin

            details[part_id] = {
                "sorted": bool(is_correct),
                "bin_side": "left" if in_left_bin else ("right" if in_right_bin else "none"),
                "is_correct": bool(is_correct),
                "semantics": sem,
            }

            if is_correct:
                current_scored_parts.add(part_id)
                if len(current_scored_parts) >= int(max_parts):
                    break

        score = min(float(score_per_part) * int(max_parts), len(current_scored_parts) * float(score_per_part))
        return int(score), current_scored_parts, details
    except Exception as e:
        raise Exception(f"task2_check_parts_in_correct_bin 异常: {e}")


def task2_calculate_total_score(
    grab_score: int,
    sort_score: int,
    max_grab_score: int = 40,
    max_sort_score: int = 40,
) -> dict:
    """
    Task2 总分汇总（官方标准）。

    官方评分标准：
    - 跟随：20分，8个工件，每个2.5分
    - 抓取：40分，8个工件，每个5分
    - 放置/分拣：40分，8个工件，每个5分

    注意：
    当前函数接口只有 grab_score 和 sort_score，没有 follow_score 参数。
    因此本函数只汇总“抓取 + 放置/分拣”两项，跟随分应由外层总评逻辑额外相加。
    """
    grab_score = int(max(0, min(int(max_grab_score), int(grab_score))))
    sort_score = int(max(0, min(int(max_sort_score), int(sort_score))))

    total_score = int(max(0, min(100, grab_score + sort_score)))

    return {
        "grab_count": int(grab_score / 5),
        "sort_count": int(sort_score / 5),
        "grab_score": grab_score,
        "sort_score": sort_score,
        "total_score": total_score,
        "max_grab_score": int(max_grab_score),
        "max_sort_score": int(max_sort_score),
    }


def task2_check_end_effector_followed_parts(
    parts_poses_dict: Any,
    ee_poses: dict[str, Any],
    scored_parts: Iterable[str] | None = None,
    distance_threshold: float = 0.10,
    score_per_part: float = 2.5,
    max_parts: int = 8,
) -> tuple[float, set[str], dict]:
    """Task2 跟随评分（官方标准，满分 20，每个零件 2.5 分）。

    任一端执行器到达零件 10 cm 范围内即计分，每个零件只计一次。

    Args:
        parts_poses_dict: 零件位姿列表。
        ee_poses: 端执行器位姿字典，值需包含至少 3 个位置分量。
        scored_parts: 已计分的零件 ID 集合。
        distance_threshold: 跟随距离阈值（米），默认 0.10。
        score_per_part: 每个零件得分，默认 2.5。
        max_parts: 最大计分零件数，默认 8。

    Returns:
        (score, scored_parts, details): 跟随总分、已计分集合、每个零件的距离详情。
    """
    current_scored_parts: set[str] = set(scored_parts or [])
    details: dict[str, dict] = {}
    ee_points: list[np.ndarray] = []
    for value in ee_poses.values():
        vec = safe_vec3(value)
        if vec is not None:
            ee_points.append(vec)

    for idx, part in enumerate(iter_part_dicts(parts_poses_dict)):
        part_id = _part_id(part, idx)
        pos = safe_vec3(part.get("position"))
        if pos is None or not ee_points:
            continue
        distances = [float(np.linalg.norm(pos - ee_pos)) for ee_pos in ee_points]
        min_distance = min(distances)
        followed = min_distance <= float(distance_threshold)
        details[part_id] = {"followed": followed, "min_distance": min_distance}
        if followed and part_id not in current_scored_parts:
            current_scored_parts.add(part_id)
            if len(current_scored_parts) >= int(max_parts):
                break

    score = min(float(score_per_part) * int(max_parts), len(current_scored_parts) * float(score_per_part))
    return score, current_scored_parts, details


# ------------------------task3评估工具------------------------------
def task3_check_insertion_success(
    parts_poses_dict: Any,
    foam_pose: np.ndarray,  # 泡棉的实时位姿 [x, y, z, qx, qy, qz, qw]
    local_slots_offsets: dict, # 槽位相对于泡棉中心的偏移量
    dist_threshold: float = 0.025,
    height_threshold: float = 0.015,
) -> tuple[int, set[str]]:
    """
    Task3 嵌装评分优化版：基于泡棉实时位姿进行坐标变换。
    """
    from scipy.spatial.transform import Rotation as R
    
    parts = iter_part_dicts(parts_poses_dict)
    scored_parts = set()
    total_score = 0
    
    # 获取泡棉的旋转和平移
    foam_pos = foam_pose[:3]
    foam_rot = R.from_quat(foam_pose[3:]).as_matrix()
    
    for slot_id, offset in local_slots_offsets.items():
        # 计算该槽位在世界坐标系下的期望位置：W_pos = R_foam * local_offset + T_foam
        target_world_pos = foam_rot @ np.array(offset['offset']) + foam_pos
        target_type = offset['type']
        
        for part in parts:
            part_id = part.get("prim_path", "")
            if part_id in scored_parts: continue
            if _extract_semantics(part) != target_type: continue
            
            part_pos = safe_vec3(part.get("position"))
            if part_pos is None: continue
            
            # 1. 检查水平距离 (XY)
            horizontal_dist = np.linalg.norm(part_pos[:2] - target_world_pos[:2])
            # 2. 检查高度 (Z) - 必须足够低才算嵌入
            height_diff = abs(part_pos[2] - target_world_pos[2])
            
            if horizontal_dist < dist_threshold and height_diff < height_threshold:
                total_score += 15
                scored_parts.add(part_id)
                break
                
    return min(90, total_score), scored_parts

def task3_calculate_specific_score(
    parts_poses_dict: Any,
    dist_threshold: float = 0.05,
    height_threshold: float = 0.03
) -> tuple[int, set[str]]:
    """
    针对 Task3 的精确评分：
    02, 03, 04 物体 -> 第二组空槽 (Y=0.21)
    05, 06, 07 物体 -> 第一组空槽 (Y=0.41)
    """
    # 按照你的要求提供绝对坐标
    group1_slots = [np.array([0.54, 0.41, 1.04]), np.array([0.76, 0.41, 1.04]), np.array([0.98, 0.41, 1.04])]
    group2_slots = [np.array([0.54, 0.21, 1.04]), np.array([0.76, 0.21, 1.04]), np.array([0.98, 0.21, 1.04])]
    
    total_score = 0
    scored_parts = set()
    used_slots = set() 

    parts = list(iter_part_dicts(parts_poses_dict))
    
    for part in parts:
        prim_path = part.get("prim_path", "")
        pos = safe_vec3(part.get("position"))
        if pos is None: continue
        
        target_slots = []
        slot_prefix = ""
        # 判定归属组别
        if any(x in prim_path for x in ["05", "06", "07"]):
            target_slots = group1_slots
            slot_prefix = "G1_"
        elif any(x in prim_path for x in ["02", "03", "04"]):
            target_slots = group2_slots
            slot_prefix = "G2_"
        else:
            continue # 如果都不是，跳过该物体不计分
        
        for i, slot_pos in enumerate(target_slots):
            slot_id = f"{slot_prefix}{i}"
            if slot_id in used_slots: continue
            
            # 仅计算 XY 平面距离
            h_dist = np.linalg.norm(pos[:2] - slot_pos[:2])
            # 计算高度误差
            v_dist = abs(pos[2] - slot_pos[2])
            
            # 💡 修正 3：使用 <= 防止正好卡在边缘距离而判定失败
            if h_dist <= dist_threshold and v_dist <= height_threshold:
                total_score += 15
                scored_parts.add(prim_path)
                used_slots.add(slot_id)
                break # 该物体已成功入槽，停止遍历其他槽
                
    return min(90, total_score), scored_parts

def task3_check_parts_inserted_in_slots(
    parts_poses_dict: Any,
    slots: list[dict],
    scored_parts: Iterable[str] | None = None,
    used_slots: Iterable[str] | None = None,
    dist_threshold: float = 0.05,
    height_threshold: float = 0.03,
    score_per_part: float = 7.5,
    max_parts: int = 6,
    require_released: bool = False,
    require_static: bool = False,
) -> tuple[float, set[str], set[str]]:
    """Task3 嵌装评分（官方标准，满分 45，每个零件 7.5 分）。

    零件按语义类别(A/B)匹配对应槽位，水平距离和高度差均在阈值内即计分。
    每槽位最多使用一次，每零件最多计分一次。

    Args:
        parts_poses_dict: 零件位姿列表。
        slots: 槽位信息列表，每项含 slot_id、type、position。
        scored_parts: 已计分的零件 ID 集合。
        used_slots: 已被占用的槽位 ID 集合。
        dist_threshold: 水平距离阈值（米），默认 0.05。
        height_threshold: 高度差阈值（米），默认 0.03。
        score_per_part: 每个零件得分，默认 7.5。
        max_parts: 最大计分零件数，默认 6。
        require_released: 是否要求零件已释放。
        require_static: 是否要求零件已静止。

    Returns:
        (score, scored_parts, used_slots): 嵌装总分、已计分零件集合、已占用槽位集合。
    """
    current_scored_parts: set[str] = set(scored_parts or [])
    current_used_slots: set[str] = set(used_slots or [])

    for idx, part in enumerate(iter_part_dicts(parts_poses_dict)):
        part_id = _part_id(part, idx)
        if part_id in current_scored_parts:
            continue
        if not _is_released(part, require_released) or not _is_static(part, require_static):
            continue
        part_pos = safe_vec3(part.get("position"))
        if part_pos is None:
            continue
        sem = _extract_semantics(part)
        for slot in slots:
            slot_id = str(slot["slot_id"])
            if slot_id in current_used_slots:
                continue
            if str(slot["type"]).upper() != sem:
                continue
            slot_pos = safe_vec3(slot["position"])
            if slot_pos is None:
                continue
            horizontal_dist = float(np.linalg.norm(part_pos[:2] - slot_pos[:2]))
            height_diff = abs(float(part_pos[2]) - float(slot_pos[2]))
            if horizontal_dist <= float(dist_threshold) and height_diff <= float(height_threshold):
                current_scored_parts.add(part_id)
                current_used_slots.add(slot_id)
                break
        if len(current_scored_parts) >= int(max_parts):
            break

    score = min(float(score_per_part) * int(max_parts), len(current_scored_parts) * float(score_per_part))
    return score, current_scored_parts, current_used_slots


def task3_time_score(elapsed: float) -> int:
    """Task3 时间评分（官方标准，满分 10）。

    360 秒内完成得满分，每超 60 秒扣 5 分，最低 0 分。
    """
    return int(calculate_time_score(
        elapsed_seconds=elapsed,
        full_score=10,
        full_time_seconds=360.0,
        penalty_interval_seconds=60.0,
        penalty_per_interval=5,
    ))


# ----------------- Task4 评估工具函数 -----------------
def task4_check_box_joints_success(
    current_box_joints: np.ndarray,
    target_box_joints: np.ndarray,
    threshold: float = 0.2
) -> tuple[bool, dict]:
    """检查盒子关节位置是否达到目标。"""
    current = np.asarray(current_box_joints, dtype=np.float32).reshape(-1)
    target = np.asarray(target_box_joints, dtype=np.float32).reshape(-1)
    errors = np.abs(current - target)
    is_success = bool(np.all(errors < threshold))

    return is_success, {
        "max_error": float(np.max(errors)),
        "mean_error": float(np.mean(errors)),
        "all_errors": errors.copy()
    }


def task4_check_short_edge_close_score(
    current_box_joints: np.ndarray,
    short_edge_targets: tuple[float, float] | list[float] | np.ndarray,
    joint_indices: tuple[int, int] = (2, 3),
    threshold: float = 0.2,
    score_per_edge: int = 15,
) -> tuple[int, dict]:
    """Task4 短边评分（总分 30，每个短边 15）。"""
    joints = np.asarray(current_box_joints, dtype=np.float32).reshape(-1)
    targets = np.asarray(short_edge_targets, dtype=np.float32).reshape(-1)

    closed: list[bool] = []
    errors: list[float] = []
    score = 0
    for i, j_idx in enumerate(joint_indices):
        err = abs(float(joints[j_idx]) - float(targets[i]))
        is_closed = err <= float(threshold)
        errors.append(err)
        closed.append(is_closed)
        if is_closed:
            score += int(score_per_edge)

    return int(min(int(score_per_edge) * len(joint_indices), score)), {
        "short_edge_closed": closed,
        "short_edge_errors": errors,
    }


def task4_check_long_edge_close_score(
    current_box_joints: np.ndarray,
    long_edge_targets: tuple[float, float] | list[float] | np.ndarray,
    joint_indices: tuple[int, int] = (0, 1),
    threshold: float = 0.2,
    score_per_edge: int = 15,
) -> tuple[int, dict]:
    """Task4 长边评分（总分 30，每个长边 15）。"""
    joints = np.asarray(current_box_joints, dtype=np.float32).reshape(-1)
    targets = np.asarray(long_edge_targets, dtype=np.float32).reshape(-1)

    closed: list[bool] = []
    errors: list[float] = []
    score = 0
    for i, j_idx in enumerate(joint_indices):
        err = abs(float(joints[j_idx]) - float(targets[i]))
        is_closed = err <= float(threshold)
        errors.append(err)
        closed.append(is_closed)
        if is_closed:
            score += int(score_per_edge)

    return int(min(int(score_per_edge) * len(joint_indices), score)), {
        "long_edge_closed": closed,
        "long_edge_errors": errors,
    }


def task4_time_score(
    elapsed_seconds: float,
    full_score: int = 40,
    full_time_seconds: float = 180.0,
    penalty_interval_seconds: float = 30.0,
    penalty_per_interval: int = 5,
) -> int:
    """Task4 时间评分（官方标准，满分 40）。

    180 秒内完成得满分，每超 30 秒扣 5 分，最低 0 分。
    """
    return int(calculate_time_score(
        elapsed_seconds=elapsed_seconds,
        full_score=full_score,
        full_time_seconds=full_time_seconds,
        penalty_interval_seconds=penalty_interval_seconds,
        penalty_per_interval=penalty_per_interval,
    ))


def task4_get_bimanual_collaboration_factor(
    is_bimanual_collaboration: bool,
    single_arm_factor: float = 0.7,
    bimanual_factor: float = 1.0,
) -> float:
    """Task4 协同系数：单臂=0.7，双臂协同=1.0。"""
    return float(bimanual_factor if is_bimanual_collaboration else single_arm_factor)


def task4_calculate_total_score(
    short_edge_score: int,
    long_edge_score: int,
    time_score: int,
) -> dict:
    """
    Task4 总分汇总（官方标准）。

    官方标准：
    - 纸箱短边是否闭合：30分
    - 纸箱长边是否闭合：30分
    - 时间分：40分
    """
    short_edge_score = int(max(0, min(30, int(short_edge_score))))
    long_edge_score = int(max(0, min(30, int(long_edge_score))))
    time_score = int(max(0, min(40, int(time_score))))

    raw_score = short_edge_score + long_edge_score + time_score

    return {
        "raw_score": int(raw_score),
        "final_score": int(raw_score),
    }
