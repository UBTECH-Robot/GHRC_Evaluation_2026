"""双容器评测共用常量。"""

from typing import Literal, TypeAlias

TaskNameLiteral: TypeAlias = Literal["TASK1", "TASK2", "TASK3", "TASK4"]

TASK_CHOICES = ["task1", "task2", "task3", "task4"]

TASK_MAX_STEPS: dict[TaskNameLiteral, int] = {
    "TASK1": 10000,
    "TASK2": 10000,
    "TASK3": 10000,
    "TASK4": 2000,
}

DEFAULT_POS_THRESHOLD = 0.5
DEFAULT_ORI_THRESHOLD = 0.5

TASK_DEFAULT_CONFIG_PATH: dict[str, str] = {
    "task1": "Ubtech_sim/config/Part_Sorting.yaml",
    "task2": "Ubtech_sim/config/Conveyor_Sorting.yaml",
    "task3": "Ubtech_sim/config/Foam_Inlaying.yaml",
    "task4": "Ubtech_sim/config/Packing_Box.yaml",
}

TASK_DEFAULT_TEXT: dict[str, str] = {
    "task1": "pick and place",
    "task2": "conveyor sorting",
    "task3": "multi-category sorting",
    "task4": "packing box",
}

TASK_DEFAULT_POLICY_PATH: dict[str, str] = {
    "task1": "challenge2026_baseline/Part_Sorting/act/pretrained_model",
    "task2": "challenge2026_baseline/Conveyor_Sorting/act/pretrained_model",
    "task3": "challenge2026_baseline/Foam_Inlaying/act/pretrained_model",
    "task4": "challenge2026_baseline/Packing_Box/act/pretrained_model",
}

TASK_DEFAULT_MAX_STEPS: dict[str, int] = {
    "task1": TASK_MAX_STEPS.get("TASK1", 10000),
    "task2": TASK_MAX_STEPS.get("TASK2", 10000),
    "task3": TASK_MAX_STEPS.get("TASK3", 10000),
    "task4": TASK_MAX_STEPS.get("TASK4", 2000),
}
