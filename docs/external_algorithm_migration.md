# 外部算法项目迁移示例

本文说明选手把非 LeRobot 项目迁移到 GHRC 评测仓库时，文件应如何放置、如何声明依赖、如何通过 `PolicyAdapter` 接入评测，以及如何用随机动作示例验证迁移链路。

自定义策略接口规范见 [GHRC 自定义策略接入指南](custom_policy.md)。完整评测运行流程见 [GHRC 评测系统使用指南](eval_guide.md)。

---

## 1. 适用范围

本文适用于以下场景：

- 策略来自另一个完整 Python 项目目录。
- 策略不是标准 LeRobot checkpoint，无法直接通过 `policy_type` + `policy_path` 加载。
- 策略需要自定义图像预处理、状态拼接、规划器、外部推理引擎或专用依赖。
- 策略最终仍能输出 GHRC 评测系统要求的一维 action。

不建议把外部项目代码直接散落到 `src/lerobot` 内部。推荐保留项目原始目录结构，只新增一个轻量 GHRC adapter 作为边界层。

---

## 2. 推荐目录结构

把外部算法作为独立项目文件夹放在仓库根目录：

```text
challengeBaseline_newFramework/
├── my_team_policy/
│   ├── my_algorithm/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── planner.py
│   │   └── checkpoint_utils.py
│   ├── checkpoints/
│   │   └── best.pt
│   ├── __init__.py
│   └── ghrc_adapter.py
├── eval_config/
│   └── eval_infer.yaml
└── src/lerobot/
```

关键要求：

| 项目    | 要求                                                                               |
| ------- | ---------------------------------------------------------------------------------- |
| import  | 项目目录必须能被 Python import                                                     |
| 包结构  | 每级 Python 包目录建议包含 `__init__.py`                                         |
| adapter | 新增 `ghrc_adapter.py`，只负责 GHRC 接口转换                                     |
| 权重    | checkpoint 放在容器内可访问路径，并通过 `policy_path` 或 `adapter_config` 指定 |
| 依赖    | 额外依赖写入 Dockerfile、requirements 或 pyproject 配置                            |

如果项目目录放在仓库根目录，并从 `/workspace/eval` 启动，根目录通常已在 `sys.path` 中。若放在更深路径，需要在 Dockerfile 或启动脚本中设置 `PYTHONPATH`。

---

## 3. 仓库内随机动作示例

仓库提供了一个外部项目迁移示例，网络最终输出随机动作。该示例只用于验证目录结构、import、adapter 加载和 action 返回链路，不代表任务策略能力。

```text
challengeBaseline_newFramework/
├── external_policy_examples/
│   └── random_action_project/
│       ├── external_algo/
│       │   ├── __init__.py
│       │   └── random_network.py
│       ├── __init__.py
│       └── ghrc_adapter.py
└── eval_config/
    └── eval_infer_external_random.yaml
```

示例文件：

| 文件                                                                               | 说明                          |
| ---------------------------------------------------------------------------------- | ----------------------------- |
| `external_policy_examples/random_action_project/external_algo/random_network.py` | 模拟外部项目中的策略网络      |
| `external_policy_examples/random_action_project/ghrc_adapter.py`                 | GHRC `PolicyAdapter` 包装层 |
| `eval_config/eval_infer_external_random.yaml`                                    | infer 配置示例                |

---

## 4. 外部网络示例

`random_network.py` 模拟一个外部项目内部的策略对象：

```python
class RandomActionNetwork:
    def forward(self, observation: dict) -> list[float]:
        return [
            self._rng.uniform(self.action_low, self.action_high)
            for _ in range(self.action_dim)
        ]
```

真实迁移时，选手可以把这里替换为自己的模型加载、图像预处理、状态编码、规划器或推理引擎。外部项目内部代码不需要知道 GHRC WebSocket 协议，只需要由 adapter 调用并返回 action。

---

## 5. GHRC Adapter 示例

`ghrc_adapter.py` 负责把 GHRC observation 转成外部项目输入，并把外部项目输出转成 GHRC action：

```python
from src.lerobot.sim_eval.policy_adapter import PolicyAdapter
from .external_algo.random_network import RandomActionNetwork


class ExternalRandomPolicyAdapter(PolicyAdapter):
    def load(self, model_path, device, config):
        self.network = RandomActionNetwork(
            action_dim=int(config.get("action_dim", 20)),
            action_low=float(config.get("action_low", -1.0)),
            action_high=float(config.get("action_high", 1.0)),
            seed=config.get("seed"),
        )

    def predict(self, observation, context):
        return self.network.forward(observation)

    def reset(self, reset_context=None):
        self.network.reset()

    def close(self):
        self.network = None
```

迁移真实项目时，adapter 中通常需要完成：

| 方法          | 迁移职责                                                                             |
| ------------- | ------------------------------------------------------------------------------------ |
| `load()`    | 读取 `policy_path` 和 `adapter_config`，加载 checkpoint、初始化模型、设置 device |
| `predict()` | 将 GHRC observation 转成外部算法输入，调用模型或规划器，返回一维 action              |
| `reset()`   | 清理 RNN hidden state、action chunk、history buffer、planner 状态等跨 episode 状态   |
| `close()`   | 释放 GPU 显存、文件句柄、推理引擎等资源                                              |

---

## 6. YAML 配置

随机动作示例配置：

```yaml
adapter_class: external_policy_examples.random_action_project.ghrc_adapter:ExternalRandomPolicyAdapter
adapter_config:
  action_dim: 20
  action_low: -1.0
  action_high: 1.0
  seed: 2026
policy_type: null
policy_path: null
```

真实项目配置示例：

```yaml
adapter_class: my_team_policy.ghrc_adapter:MyAdapter
adapter_config:
  action_dim: 20
  image_size: [224, 224]
  use_language: true
policy_type: null
policy_path: /workspace/eval/my_team_policy/checkpoints/best.pt
```

字段说明：

| 字段               | 说明                                                                     |
| ------------------ | ------------------------------------------------------------------------ |
| `adapter_class`  | `模块路径:类名`，由 `ghrc_eval_infer.py` 使用 `importlib` 动态导入 |
| `adapter_config` | 自定义参数字典，会传入 `adapter.load(model_path, device, config)`      |
| `policy_type`    | 设置 `adapter_class` 后可为 `null`，不会走 LeRobot 默认 adapter      |
| `policy_path`    | 真实模型权重路径；如果模型不需要权重，可设为 `null`                    |

---

## 7. 运行验证

### 7.1 验证 Python import

在仓库根目录或 infer 容器内执行：

```bash
python -c "from external_policy_examples.random_action_project.ghrc_adapter import ExternalRandomPolicyAdapter; print(ExternalRandomPolicyAdapter)"
```

真实项目替换为自己的 adapter：

```bash
python -c "from my_team_policy.ghrc_adapter import MyAdapter; print(MyAdapter)"
```

### 7.2 启动 infer

```bash
python -m lerobot.scripts.ghrc_eval_infer \
  --config eval_config/eval_infer_external_random.yaml \
  --task task4
```

### 7.3 启动 sim-eval

```bash
python -m lerobot.scripts.ghrc_eval_sim \
  --config eval_config/eval_sim.yaml \
  --task task4
```

随机动作策略通常不会完成任务，失败或超时是预期结果。该示例的目标是验证：

- 外部项目目录可被 import。
- `adapter_class` 可被 infer 动态加载。
- `predict()` 返回的一维 action 可被 sim 侧解码。
- episode 结束时 `reset()` 可被调用。

---

## 8. 依赖和镜像要求

外部项目若有额外依赖，必须随选手镜像一起交付。

| 类型            | 处理方式                                            |
| --------------- | --------------------------------------------------- |
| Python 依赖     | 写入 Dockerfile、requirements.txt 或 pyproject.toml |
| 系统库          | 写入 Dockerfile，避免运行时手动安装                 |
| 大模型权重      | 放入镜像、挂载目录或赛事允许的模型路径              |
| 环境变量        | 在启动脚本或容器配置中显式声明                      |
| CUDA / TensorRT | 确认与评测基础镜像版本兼容                          |

正式提交前，建议在与赛事评测一致的镜像中完成一次 `import -> infer 启动 -> sim 连接 -> episode reset` 的完整链路验证。

---

## 9. 不允许修改范围

外部项目迁移时，不应为了适配算法修改以下内容：

| 路径或逻辑                                                                      | 原因                   |
| ------------------------------------------------------------------------------- | ---------------------- |
| `src/lerobot/sim_eval` 中的通信、断言和评分逻辑                               | 影响评测一致性和公平性 |
| `src/lerobot/scripts/ghrc_eval_sim.py` 的 action 解码、episode 执行和结果生成 | 影响仿真行为和分数     |
| 任务评价阈值和成功条件                                                          | 影响赛事统一标准       |
