# GHRC 自定义策略接入指南

本文面向参赛队伍，说明如何把自定义 policy 接入 GHRC infer 容器，并保证策略能被 `sim-eval` 容器稳定调用。

上级评测流程见 [GHRC 评测系统使用指南](eval_guide.md)。如果策略来自另一个完整算法项目文件夹，请先阅读本文，再继续阅读 [外部算法项目迁移示例](external_algorithm_migration.md)。

---

## 1. 接入方式总览

| 接入方式 | 适用情况 | 需要修改 |
| --- | --- | --- |
| LeRobot 默认 adapter | checkpoint 符合 LeRobot `from_pretrained` 和 `select_action` 接口 | `eval_config/eval_infer.yaml` |
| 自定义 `PolicyAdapter` | 自定义 PyTorch、ONNX、TensorRT、RL、规划器或混合算法 | 新增 adapter 文件，并配置 `adapter_class` |
| 外部项目迁移 | 原算法是一个完整项目目录，需要保留内部结构 | 新增项目目录和 `ghrc_adapter.py`，详见 [外部算法项目迁移示例](external_algorithm_migration.md) |

正式评测中，选手应优先通过 YAML 和自定义 adapter 接入策略，不应修改通信协议、断言或评分逻辑。

---

## 2. LeRobot 默认 adapter

LeRobot 默认 adapter 适用于标准 LeRobot action policy。配置示例：

```yaml
adapter_type: lerobot
adapter_class: null
policy_type: act
policy_path: null
require_task_policy_paths: true
task_policy_paths:
  task1: ../challenge2026_baseline/task1/act/pretrained_model
  task2: ../challenge2026_baseline/task2/act/pretrained_model
  task3: ../challenge2026_baseline/task3/act/pretrained_model
  task4: ../challenge2026_baseline/task4/act/pretrained_model
```

加载链路：

```text
eval_infer.yaml
  -> LeRobotPolicyAdapter.load()
    -> load_policy(model_path, policy_type, device)
      -> get_policy_class(policy_type)
      -> PolicyClass.from_pretrained(model_path)
      -> policy.select_action(observation_batch)
```

### 2.1 支持边界

当前默认 adapter 并不等价于“支持所有 LeRobot policy”。直接可用需要同时满足以下条件：

| 条件 | 要求 |
| --- | --- |
| policy class | `policy_type` 能被 LeRobot `get_policy_class()` 找到 |
| checkpoint | checkpoint 能被 `config_class.from_pretrained()` 和 `policy_cls.from_pretrained()` 加载 |
| 推理接口 | policy 实现可用的 `select_action(batch)` |
| observation | 当前评测 observation key 与 checkpoint 的输入 feature 匹配 |
| action | 输出是一维动作向量，维度能被 sim 侧解码 |

以下情况建议改用自定义 `PolicyAdapter`：

- policy 需要额外 tokenizer、processor、language token 或专用图像预处理。
- policy 不是动作策略，例如 reward classifier。
- policy 的输入 key、状态拼接方式、相机命名或 action 后处理与当前评测环境不一致。
- 迁移的是非 LeRobot 框架、外部项目或自研推理引擎。

---

## 3. 自定义 PolicyAdapter

自定义 adapter 是 GHRC 推荐的通用扩展方式。它负责把评测系统传入的 observation 转成选手模型输入，并返回一维 action。

### 3.1 接口定义

```python
from src.lerobot.sim_eval.policy_adapter import PolicyAdapter, InferenceContext, ResetContext


class MyPolicyAdapter(PolicyAdapter):
    def load(self, model_path: str, device: str, config: dict) -> None:
        """加载模型、权重、推理引擎和自定义配置。"""
        ...

    def predict(self, observation: dict, context: InferenceContext):
        """根据当前 observation 返回一维 action。"""
        ...

    def reset(self, reset_context: ResetContext | None = None) -> None:
        """episode 结束后清理跨步状态。"""
        ...

    def close(self) -> None:
        """释放模型、显存、文件句柄或推理引擎资源。"""
        ...
```

接口职责：

| 方法 | 调用时机 | 必要要求 |
| --- | --- | --- |
| `load()` | infer 服务启动时调用一次 | 加载模型并进入推理模式，失败时抛出清晰异常 |
| `predict()` | 每个仿真 step 调用 | 返回一维 `torch.Tensor`、`np.ndarray` 或 `list[float]` |
| `reset()` | episode 结束或任务重置时调用 | 清理 RNN hidden state、action chunk、history buffer、planner 状态等 |
| `close()` | infer 服务退出时调用 | 释放显存、文件句柄、TensorRT engine 等资源 |

### 3.2 YAML 配置

```yaml
adapter_type: lerobot
adapter_class: my_team_policy.ghrc_adapter:MyPolicyAdapter
adapter_config:
  action_dim: 20
  image_size: [224, 224]
  normalize_state: true

policy_type: null
policy_path: /workspace/eval/my_team_policy/checkpoints/best.pt
```

字段说明：

| 字段 | 是否必填 | 说明 |
| --- | --- | --- |
| `adapter_class` | 是 | 自定义 adapter 类路径，格式为 `module.path:ClassName` 或 `module.path.ClassName` |
| `adapter_config` | 否 | 原样传入 `load(model_path, device, config)` 的自定义字典 |
| `policy_path` | 视策略而定 | 权重路径或模型目录；自定义 adapter 可以按需使用 |
| `policy_type` | 否 | 设置 `adapter_class` 后不走 LeRobot 默认加载逻辑，可设为 `null` |

`adapter_class` 指向的类必须继承 `PolicyAdapter`，否则 infer 启动会失败。

---

## 4. 最小示例：零动作 policy

仓库提供了一个可直接运行的最小自定义 policy，用于验证配置、导入、WebSocket 通信和 action 解码链路：

| 项目 | 路径 |
| --- | --- |
| 示例代码 | `src/lerobot/sim_eval/zero_action_policy.py` |
| 示例配置 | `eval_config/eval_infer_zero_action.yaml` |

核心逻辑：

```python
class ZeroActionPolicyAdapter(PolicyAdapter):
    def load(self, model_path: str, device: str, config: dict) -> None:
        self.action_dim = int(config.get("action_dim", 20))
        self._action = [0.0] * self.action_dim

    def predict(self, observation: dict, context: InferenceContext) -> list[float]:
        return list(self._action)
```

YAML：

```yaml
adapter_class: src.lerobot.sim_eval.zero_action_policy:ZeroActionPolicyAdapter
adapter_config:
  action_dim: 20
policy_type: null
policy_path: null
```

infer 容器内运行：

```bash
python -m lerobot.scripts.ghrc_eval_infer \
  --config eval_config/eval_infer_zero_action.yaml \
  --task task4
```

零动作策略通常不会完成任务。该示例只用于验证评测链路是否可以正常启动、连接和返回动作。

---

## 5. 外部项目迁移入口

如果选手迁移的是另一个项目文件夹，例如自研 RL 项目、视觉语言模型项目或已有机器人算法仓库，推荐保留原项目目录结构，只新增一个 `ghrc_adapter.py` 作为 GHRC 接口层。

完整随机动作外部项目示例见 [外部算法项目迁移示例](external_algorithm_migration.md)：

| 项目 | 路径 |
| --- | --- |
| 示例项目 | `external_policy_examples/random_action_project/` |
| 示例配置 | `eval_config/eval_infer_external_random.yaml` |
| 迁移说明 | [external_algorithm_migration.md](external_algorithm_migration.md) |

该示例保留外部项目自己的 `external_algo/random_network.py`，并通过 `ghrc_adapter.py` 输出随机动作，用于验证以下内容：

- 外部项目目录可以被 Python import。
- `adapter_class` 可以正确加载自定义 adapter。
- `predict()` 返回的一维 action 可以被 sim 侧解码。
- episode 结束时 `reset()` 可以被正常调用。

---

## 6. Observation 格式

`predict()` 收到的 `observation` 是字典格式，常见 key 如下：

```python
{
    "observation.state": torch.Tensor,
    "observation.images.cam_high": torch.Tensor,
    "observation.images.cam_left_wrist": torch.Tensor,
    "observation.images.cam_right_wrist": torch.Tensor,
}
```

注意事项：

- state 通常是一维机器人状态向量。
- image 通常是 `torch.Tensor`，图像 key 以实际任务配置和仿真输出为准。
- 自定义 adapter 应在 `predict()` 内完成选手模型所需的裁剪、归一化、resize、相机重命名和 batch 维度处理。
- 不要假设所有 policy 的输入格式相同，尤其是需要 language token 或多模态 processor 的模型。

---

## 7. Action 输出格式

`predict()` 返回值支持以下类型：

| 类型 | 处理方式 |
| --- | --- |
| `torch.Tensor` | 自动 `detach()`、转 float、转 CPU 并展平为一维 |
| `np.ndarray` | 自动转为 `torch.Tensor` |
| `list[float]` | 自动转为 `torch.Tensor` |

action 要求：

- 必须是一维动作向量。
- 推荐输出 20 维 action。
- `task4` 输出 18 维时，sim 侧会在右侧补 2 维夹爪控制量。
- 不应返回 batch 维、字典、嵌套列表或未归一化到策略约定范围之外的值。

---

## 8. 不允许修改范围

正式评测提交中，不应为了适配策略修改以下内容：

| 路径 | 原因 |
| --- | --- |
| `src/lerobot/sim_eval` 中的通信协议、断言、评分逻辑 | 影响评测一致性 |
| `src/lerobot/scripts/ghrc_eval_sim.py` 的 action 解码和结果生成逻辑 | 影响仿真执行和评分结果 |
| 任务评价阈值和成功条件 | 影响赛事公平性 |

如确需扩展策略加载方式，应优先新增自定义 `PolicyAdapter`，或在选手项目目录中新增包装层。

---

## 9. 接入检查清单

| 检查项 | 要求 |
| --- | --- |
| 配置文件 | `eval_config/eval_infer.yaml` 或独立示例 YAML 可被 infer 读取 |
| import | `python -c "from my_team_policy.ghrc_adapter import MyAdapter"` 能成功 |
| 继承关系 | 自定义类继承 `PolicyAdapter` |
| `load()` | 能加载模型、权重和配置，并进入 eval 推理模式 |
| `predict()` | 输入任意合法 observation 时返回一维 action |
| action 维度 | 推荐 20 维；task4 可 18 维 |
| `reset()` | episode 结束后清理跨步状态 |
| 依赖 | 额外 Python 包、系统库和权重文件已写入镜像或挂载路径 |
| 日志 | 加载失败、权重缺失、维度错误时输出清晰错误 |

完成上述检查后，再使用 [GHRC 评测系统使用指南](eval_guide.md) 中的本地评测流程启动完整评测。
