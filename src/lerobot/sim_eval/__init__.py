"""Simulation evaluation modules — 双容器隔离评测。

sim-eval 侧：仿真、观测采集、断言评测、结果落盘
infer 侧：模型加载、推理、policy reset

通信：WebSocket control(8765) / stream(8766)，msgpack 序列化 + lz4 压缩。
"""

from __future__ import annotations

from .const import (
    DEFAULT_ORI_THRESHOLD,
    DEFAULT_POS_THRESHOLD,
    TASK_CHOICES,
    TASK_DEFAULT_CONFIG_PATH,
    TASK_DEFAULT_MAX_STEPS,
    TASK_DEFAULT_POLICY_PATH,
    TASK_DEFAULT_TEXT,
    TASK_MAX_STEPS,
    TaskNameLiteral,
)
from .container_config import load_container_config, require_config_keys

__all__ = [
    "TASK_CHOICES",
    "TASK_MAX_STEPS",
    "TASK_DEFAULT_TEXT",
    "TASK_DEFAULT_MAX_STEPS",
    "TASK_DEFAULT_POLICY_PATH",
    "TASK_DEFAULT_CONFIG_PATH",
    "DEFAULT_POS_THRESHOLD",
    "DEFAULT_ORI_THRESHOLD",
    "TaskNameLiteral",
    "load_container_config",
    "require_config_keys",
]

try:
    from .common import (
        PartPoseDict,
        WorkspaceLimits,
        action_to_dict,
        build_robot,
        build_workspace_limits,
        flatten_obs,
        load_policy,
        obs_to_tensor,
        resolve_device,
    )
    from .results import (
        EpisodeResult,
        InferenceSummary,
        print_summary,
        save_episode,
        save_summary,
    )
    from .policy_adapter import (
        InferenceContext,
        LeRobotPolicyAdapter,
        PolicyAdapter,
        ResetContext,
        create_policy_adapter,
        load_adapter_class,
    )
    from .zero_action_policy import ZeroActionPolicyAdapter
    from .scoring import (
        calculate_time_score,
        check_parts_lifted_from_initial_height,
        task1_check_parts_in_box,
        task1_time_out_check,
        task2_calculate_total_score,
        task2_check_end_effector_followed_parts,
        task2_check_parts_grabbed,
        task2_check_parts_in_correct_bin,
        task3_calculate_specific_score,
        task3_check_insertion_success,
        task3_check_parts_inserted_in_slots,
        task3_time_score,
        task4_calculate_total_score,
        task4_check_box_joints_success,
        task4_check_long_edge_close_score,
        task4_check_short_edge_close_score,
        task4_time_score,
    )
    from .task_assert import (
        TASK_ASSERTION_REGISTRY,
        EpisodePartsTracker,
        NoOpAssertion,
        Task1Assertion,
        Task2Assertion,
        Task3Assertion,
        Task4Assertion,
        Task4EpisodeActionTracker,
        TaskAssertion,
        create_task_assertion,
        get_robot_ee_poses,
    )
    from .task_eval_config import (
        build_assertion_args_from_task_yaml,
        load_task_yaml_config,
    )
    from .terminal import (
        check_step_terminal,
        task1_check_parts_out_of_workspace,
        task2_check_all_parts_lost,
        task3_check_terminal,
        task4_check_box_poses_terminal,
    )

    __all__.extend(
        [
            "PartPoseDict",
            "WorkspaceLimits",
            "build_robot",
            "flatten_obs",
            "action_to_dict",
            "resolve_device",
            "load_policy",
            "obs_to_tensor",
            "build_workspace_limits",
            "check_step_terminal",
            "task1_check_parts_out_of_workspace",
            "task2_check_all_parts_lost",
            "task3_check_terminal",
            "task4_check_box_poses_terminal",
            "calculate_time_score",
            "check_parts_lifted_from_initial_height",
            "task1_check_parts_in_box",
            "task1_time_out_check",
            "task2_check_end_effector_followed_parts",
            "task2_check_parts_grabbed",
            "task2_check_parts_in_correct_bin",
            "task2_calculate_total_score",
            "task3_time_score",
            "task3_calculate_specific_score",
            "task3_check_insertion_success",
            "task3_check_parts_inserted_in_slots",
            "task4_check_box_joints_success",
            "task4_check_short_edge_close_score",
            "task4_check_long_edge_close_score",
            "task4_time_score",
            "task4_calculate_total_score",
            "TaskAssertion",
            "NoOpAssertion",
            "Task1Assertion",
            "Task2Assertion",
            "Task3Assertion",
            "Task4Assertion",
            "TASK_ASSERTION_REGISTRY",
            "Task4EpisodeActionTracker",
            "EpisodePartsTracker",
            "create_task_assertion",
            "get_robot_ee_poses",
            "build_assertion_args_from_task_yaml",
            "load_task_yaml_config",
            "EpisodeResult",
            "InferenceSummary",
            "save_episode",
            "save_summary",
            "print_summary",
            "InferenceContext",
            "ResetContext",
            "PolicyAdapter",
            "LeRobotPolicyAdapter",
            "create_policy_adapter",
            "load_adapter_class",
            "ZeroActionPolicyAdapter",
        ]
    )
except ModuleNotFoundError:
    # 允许在轻量环境中仅使用协议/配置模块。
    pass
