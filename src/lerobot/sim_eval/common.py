from __future__ import annotations

from typing import Any, TypeAlias
from pathlib import Path
import yaml
from .const import TASK_DEFAULT_CONFIG_PATH
import numpy as np
from src.lerobot.robots.walker_s2_sim.walkers2sim import WalkerS2sim
from src.lerobot.robots.walker_s2_sim.walkers2simConfig import WalkerS2Config
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


PartPoseDict: TypeAlias = dict[str, Any]
WorkspaceLimits: TypeAlias = dict[str, tuple[float, float]]


def ensure_vector(
    value: Any,
    min_size: int,
    dtype: Any = np.float32,
) -> np.ndarray | None:
    """Convert input to a 1D ndarray and validate minimal length."""
    try:
        arr = np.asarray(value, dtype=dtype).reshape(-1)
        if arr.size < int(min_size):
            return None
        return arr
    except Exception:
        return None


def safe_vec3(value: Any) -> np.ndarray | None:
    """Convert input to a vector with at least 3 values and return xyz."""
    vec = ensure_vector(value=value, min_size=3, dtype=np.float32)
    if vec is None:
        return None
    return vec[:3]


def iter_part_dicts(parts_poses_dict: Any) -> list[PartPoseDict]:
    """Normalize part poses input into a list of dict objects."""
    if isinstance(parts_poses_dict, list):
        return [item for item in parts_poses_dict if isinstance(item, dict)]
    if isinstance(parts_poses_dict, dict):
        return [parts_poses_dict]
    return []


def _make_json_safe(data: Any) -> Any:
    """递归将 numpy 数组等不可序列化类型转换为 Python 原生类型。"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, dict):
        return {k: _make_json_safe(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_make_json_safe(i) for i in data]
    if isinstance(data, (np.integer,)):
        return int(data)
    if isinstance(data, (np.floating,)):
        return float(data)
    return data


def build_robot(task: str, task_config_path: str | None) -> WalkerS2sim:
    """根据任务名和配置路径构建机器人。"""
    repo_root = Path.cwd()
    config_path = task_config_path or TASK_DEFAULT_CONFIG_PATH[task]
    abs_config = Path(config_path) if Path(config_path).is_absolute() else repo_root / config_path

    robot_cfg = WalkerS2Config()
    robot_cfg.task_name = task
    robot_cfg.task_cfg_path = str(abs_config.resolve())

    with open(robot_cfg.task_cfg_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    yaml_dir = abs_config.parent
    if "root_path" in cfg_dict:
        cfg_dict["root_path"] = str((yaml_dir / cfg_dict["root_path"]).resolve())
    robot_cfg.task_cfg = cfg_dict

    return WalkerS2sim(robot_cfg)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退到 CPU")
        return "cpu"
    return device


def load_policy(policy_path: str, policy_type: str, device: str) -> Any:
    """统一策略加载：通过 factory 支持 act / pi0 及未来扩展。"""
    from src.lerobot.policies.factory import get_policy_class

    path_obj = Path(policy_path).expanduser().absolute()
    is_local = path_obj.is_dir()

    policy_cls = get_policy_class(policy_type)
    config_cls = policy_cls.config_class
    cfg = config_cls.from_pretrained(str(path_obj), local_files_only=is_local)
    cfg.pretrained_path = str(path_obj)
    cfg.device = device

    policy = policy_cls.from_pretrained(
        pretrained_name_or_path=str(path_obj),
        config=cfg,
        local_files_only=is_local,
    )
    policy = policy.to(torch.device(device))
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()
    return policy


def obs_to_tensor(obs: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    """将观测字典转为 float32 Tensor 并添加 batch 维度。"""
    result: dict[str, torch.Tensor] = {}
    for key, value in obs.items():
        is_image = key.startswith("observation.images.")
        if isinstance(value, torch.Tensor):
            t = value
        elif isinstance(value, np.ndarray):
            t = torch.from_numpy(value)
        elif isinstance(value, (list, tuple)):
            # WebSocket JSON 反序列化后，ndarray 会变成 list。
            t = torch.from_numpy(np.asarray(value))
        else:
            raise ValueError(f"不支持的观测类型 [{key}]: {type(value)}")
        if is_image and t.dtype == torch.uint8:
            t = t.to(device=device, dtype=torch.float32, non_blocking=device.startswith("cuda")) / 255.0
        else:
            t = t.to(device=device, dtype=torch.float32, non_blocking=device.startswith("cuda"))
        # 图像从 (H, W, C) 转为 (C, H, W)
        if is_image and t.dim() == 3 and t.shape[-1] in (1, 3, 4):
            t = t.permute(2, 0, 1)
        if key == "observation.state" and t.dim() == 1:
            t = t.unsqueeze(0)
        elif is_image and t.dim() == 3:
            t = t.unsqueeze(0)
        result[key] = t
    return result


def build_workspace_limits(cfg: dict[str, Any]) -> dict[str, tuple[float, float]]:
    limits = cfg.get("workspace_limits") if isinstance(cfg, dict) else None
    if isinstance(limits, dict) and all(k in limits for k in ("x", "y", "z")):
        try:
            return {
                "x": (float(limits["x"][0]), float(limits["x"][1])),
                "y": (float(limits["y"][0]), float(limits["y"][1])),
                "z": (float(limits["z"][0]), float(limits["z"][1])),
            }
        except Exception as e:
            logger.warning(f"构建工作空间限制异常：{e}")

    scatter = cfg.get("grasp", {}).get("scatter_area", {}) if isinstance(cfg, dict) else {}
    center = scatter.get("center", [0.75, 0.28, 1.04])
    size = [0.75, 0.30]
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    sx, sy = float(size[0]), float(size[1])
    return {
        "x": (cx - sx, cx + sx),
        "y": (cy - sy, cy + sy),
        "z": (cz - 0.10, cz + 0.10),
    }


# ── WalkerS2sim 观测/动作 格式转换 ──────────────────────────

def flatten_obs(obs: dict) -> dict:
    """将 WalkerS2sim 扁平观测转为 LeRobot 标准 key 格式。

    observation.state 仅取关节位置 + 夹爪（14arm + 4finger + 2gripper = 20D），
    不包含 env_state（物体位姿），因为模型训练时未使用这些特征。
    """
    state_keys = []
    image_keys = []
    for key in obs:
        if key.startswith("observation.images.") or key in ("head_left", "head_right", "wrist_left", "wrist_right"):
            image_keys.append(key)
        elif key.endswith(".pos") or key in ("left_gripper", "right_gripper"):
            state_keys.append(key)

    result = {}
    if state_keys:
        parts = [obs[k].flatten() if isinstance(obs[k], torch.Tensor) else torch.tensor(obs[k], dtype=torch.float32).flatten() for k in sorted(state_keys)]
        result["observation.state"] = torch.cat(parts)

    cam_map = {"head_left": "observation.images.head_left", "head_right": "observation.images.head_right",
               "wrist_left": "observation.images.wrist_left", "wrist_right": "observation.images.wrist_right"}
    for key in image_keys:
        result[cam_map.get(key, key)] = obs[key]
    return result


def action_to_dict(action: torch.Tensor) -> dict:
    """将 20D action tensor 转为 WalkerS2sim.send_action 所需的扁平字典。"""
    arm_names = [
        "L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
        "L_elbow_roll_joint", "L_elbow_yaw_joint", "L_wrist_pitch_joint", "L_wrist_roll_joint",
        "R_shoulder_pitch_joint", "R_shoulder_roll_joint", "R_shoulder_yaw_joint",
        "R_elbow_roll_joint", "R_elbow_yaw_joint", "R_wrist_pitch_joint", "R_wrist_roll_joint",
    ]
    finger_names = ["L_finger1_joint", "L_finger2_joint", "R_finger1_joint", "R_finger2_joint"]
    action = action.flatten().cpu()
    result = {}
    for i, name in enumerate(arm_names):
        result[f"{name}.pos"] = action[i].item()
    for i, name in enumerate(finger_names):
        result[f"{name}.pos"] = action[14 + i].item()
    if action.numel() >= 20:
        result["left_gripper"] = action[18].item()
        result["right_gripper"] = action[19].item()
    else:
        result["left_gripper"] = 0.0
        result["right_gripper"] = 0.0
    return result
