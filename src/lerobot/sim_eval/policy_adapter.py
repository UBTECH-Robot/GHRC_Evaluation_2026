"""Policy Adapter 抽象与默认实现。

本文件定义 `infer` 容器统一使用的策略适配器接口，要求接入评测的算法实现
加载、预测、reset 与资源释放四类能力。

使用示例:

```python
from lerobot.sim_eval.policy_adapter import create_policy_adapter, InferenceContext

adapter = create_policy_adapter(adapter_type="lerobot")
adapter.load(
    model_path="/path/to/checkpoint",
    device="cuda:0",
    config={"policy_type": "act", "task": "task4", "task_text": "packing box"},
)
action = adapter.predict(
    observation={"observation.state": [0.0] * 20},
    context=InferenceContext(task="task4", task_text="packing box", episode_id=0, step=0, timestamp=0.0),
)
adapter.reset()
adapter.close()
```
"""

from __future__ import annotations

import abc
import importlib
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .common import load_policy, obs_to_tensor
from src.lerobot.policies.factory import make_pre_post_processors

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class InferenceContext:
    """推理阶段上下文。

    属性说明:
        task: 当前任务名。
        task_text: 当前任务自然语言描述。
        episode_id: 当前 episode 编号。
        step: 当前 step 编号。
        timestamp: obs 采样时间戳。
    """

    task: str
    task_text: str
    episode_id: int
    step: int
    timestamp: float

@dataclass(slots=True)
class ResetContext:
    """reset 阶段上下文。

    属性说明:
        episode_id: episode 编号。
        status: episode 结束状态。
        reason: reset 原因说明。
        metrics: episode 指标字典。
        step: 结束时 step。
    """

    episode_id: int
    status: str = "reset"
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    step: int = 0

class PolicyAdapter(abc.ABC):
    """统一策略适配器抽象基类。"""

    @abc.abstractmethod
    def load(self, model_path: str, device: str, config: dict[str, Any]) -> None:
        """加载模型和运行时配置。

        Args:
            model_path: 模型权重或模型目录路径。
            device: 运行设备。
            config: 适配器额外配置。
        """

    @abc.abstractmethod
    def predict(
        self,
        observation: dict[str, Any],
        context: InferenceContext,
    ) -> torch.Tensor | np.ndarray | list[float]:
        """基于 observation 与上下文预测动作。"""

    @abc.abstractmethod
    def reset(self, reset_context: ResetContext | None = None) -> None:
        """在 episode 结束时重置策略内部状态。"""

    @abc.abstractmethod
    def close(self) -> None:
        """释放适配器持有的资源。"""

class LeRobotPolicyAdapter(PolicyAdapter):
    """默认的 LeRobot 策略适配器。

    类属性:
        policy: 已加载的 LeRobot policy 对象。
        policy_device: policy 所在设备。
        policy_type: 策略类型，例如 act / pi0 / diffusion。
        task: 当前任务名。
        task_text: 当前任务描述。
    """

    def __init__(self) -> None:
        self.policy: Any | None = None
        self.policy_device: torch.device | None = None
        self.policy_type = ""
        self.task = ""
        self.task_text = ""
        #增加preprocessor/postprocessor属性，方便后续扩展不同策略的预处理和后处理逻辑
        self.preprocessor = None
        self.postprocessor = None

    def load(self, model_path: str, device: str, config: dict[str, Any]) -> None:
        """加载 LeRobot policy。

        Args:
            model_path: policy 权重目录。
            device: 运行设备。
            config: 至少包含 `policy_type`，可选 `task` 与 `task_text`。
        """

        policy_type = str(config.get("policy_type", "")).lower()
        if not policy_type:
            raise ValueError("LeRobotPolicyAdapter 缺少必要配置：policy_type")

        self.policy_type = policy_type
        self.task = str(config.get("task", "")).lower()
        self.task_text = str(config.get("task_text", ""))
        #加载policy的同时创建对应的预处理器和后处理器，确保它们与policy配置一致
        self.policy = load_policy(model_path, policy_type, device)
        self.policy_device = next(self.policy.parameters()).device
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=model_path,
            preprocessor_overrides={
                "device_processor": {"device": str(self.policy_device)},
            },
            postprocessor_overrides={
                "device_processor": {"device": "cpu"},
            },
        )
        
        logger.info("LeRobot policy 已加载，device=%s", self.policy_device)

    def predict(
        self,
        observation: dict[str, Any],
        context: InferenceContext,
    ) -> torch.Tensor:
        """执行一次 LeRobot 推理。

        Args:
            observation: 完整 observation 字典。
            context: 推理上下文。

        Returns:
            torch.Tensor: 动作向量。
        """

        if self.policy is None or self.policy_device is None:
            raise RuntimeError("policy 尚未加载")
        
        # 二开提供的原来版本的逻辑
        # obs_tensor = obs_to_tensor(observation, str(self.policy_device))
        # self._trim_state_to_policy(obs_tensor)
        # obs_tensor["task"] = [context.task_text or self.task_text]

        # action = self.policy.select_action(obs_tensor)
        # action = action.to(self.policy_device)
        # if action.dim() == 2 and action.shape[0] == 1:
        #     action = action.squeeze(0)
        # return action.detach().float().cpu()
        obs_tensor = obs_to_tensor(observation, str(self.policy_device))
        self._trim_state_to_policy(obs_tensor)
        obs_tensor["task"] = context.task_text or self.task_text

        batch = self.preprocessor(obs_tensor)
        action = self.policy.select_action(batch)
        action = self.postprocessor(action)

        if action.dim() == 2 and action.shape[0] == 1:
            action = action.squeeze(0)
        return action.detach().float().cpu()

    def reset(self, reset_context: ResetContext | None = None) -> None:
        """重置 LeRobot policy 内部缓存队列。

        Args:
            reset_context: reset 上下文，当前默认仅用于日志。
        """

        del reset_context
        # 增加preprocessor/postprocessor的reset逻辑，确保它们在每个episode开始时都能清理状态   
        if self.preprocessor is not None:
            self.preprocessor.reset()
        if self.postprocessor is not None:
            self.postprocessor.reset()

        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()
        
    def close(self) -> None:
        """释放 adapter 引用的模型对象。"""

        self.policy = None
        self.policy_device = None

    def _trim_state_to_policy(self, observation: dict[str, torch.Tensor]) -> None:
        """根据 policy 期望维度裁剪 observation.state。

        Args:
            observation: tensor 化后的 observation 字典。
        """

        if self.policy is None:
            return
        state = observation.get("observation.state")
        if state is None:
            return

        expected_dim: int | None = None
        input_features = getattr(getattr(self.policy, "config", None), "input_features", None)
        if isinstance(input_features, dict):
            state_feature = input_features.get("observation.state")
            expected_shape = getattr(state_feature, "shape", None)
            if expected_shape:
                expected_dim = int(expected_shape[0])

        if expected_dim is not None and state.shape[-1] > expected_dim:
            observation["observation.state"] = state[..., :expected_dim]

def load_adapter_class(class_path: str) -> type[PolicyAdapter]:
    """动态加载自定义 adapter 类。

    Args:
        class_path: 支持 `pkg.module:ClassName` 或 `pkg.module.ClassName`。

    Returns:
        type[PolicyAdapter]: 适配器类对象。
    """

    module_name: str
    class_name: str

    if ":" in class_path:
        module_name, class_name = class_path.split(":", maxsplit=1)
    else:
        module_name, _, class_name = class_path.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"非法 adapter class 路径：{class_path}")

    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    if not issubclass(adapter_cls, PolicyAdapter):
        raise TypeError(f"{class_path} 不是 PolicyAdapter 子类")
    return adapter_cls

def create_policy_adapter(
    adapter_type: str = "lerobot",
    adapter_class: str | None = None,
) -> PolicyAdapter:
    """创建策略适配器实例。

    Args:
        adapter_type: 内置适配器类型，当前支持 `lerobot`。
        adapter_class: 自定义适配器类路径，优先级高于 adapter_type。

    Returns:
        PolicyAdapter: 适配器实例。
    """

    if adapter_class:
        return load_adapter_class(adapter_class)()

    normalized_type = adapter_type.lower().strip()
    if normalized_type == "lerobot":
        return LeRobotPolicyAdapter()
    raise ValueError(f"不支持的 adapter_type：{adapter_type}")
