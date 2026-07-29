# GHRC 评测系统使用指南

本文面向赛事评测管理员、参赛队伍和本地复现人员，说明 GHRC 评测系统的运行方式、配置边界、自定义策略接入入口和自动化编排流程。

评测系统采用双容器隔离架构：`infer` 容器负责加载选手策略并通过 WebSocket 提供动作推理服务，`sim-eval` 容器负责启动 Isaac Sim、执行任务、采集结果和生成评分日志。两个容器之间使用 WebSocket 传输 msgpack + lz4 编码的观测与动作数据。

---

## 1. 文档导航

| 文档 | 适用场景 |
| --- | --- |
| 本文档 | 评测系统部署、配置、运行、故障排查和自动化编排 |
| [自定义策略接入指南](custom_policy.md) | 选手需要接入自定义 policy、修改 `adapter_class`、验证 action/observation 接口 |
| [外部算法项目迁移示例](external_algorithm_migration.md) | 选手把已有算法项目文件夹迁移到评测仓库，并通过 `PolicyAdapter` 包装接入 |

自定义策略接入请优先阅读 [自定义策略接入指南](custom_policy.md)。如果策略代码来自另一个完整项目目录，再继续阅读 [外部算法项目迁移示例](external_algorithm_migration.md)。

---

## 2. 系统职责边界

| 模块 | 容器 | 主要职责 | 是否允许选手修改 |
| --- | --- | --- | --- |
| `ghrc_eval_infer.py` | infer | 加载配置、加载 policy、启动 WebSocket 推理服务 | 原则上不建议修改；优先通过 YAML 和 `adapter_class` 扩展 |
| `PolicyAdapter` | infer | 将评测 observation 转换为选手模型输入，并返回 action | 允许新增自定义 adapter |
| `ghrc_eval_sim.py` | sim-eval | 启动仿真、连接 infer、执行 episode、解码 action | 不建议修改 |
| `src/lerobot/sim_eval` | infer / sim-eval | 通信协议、容器配置、断言、评分相关公共逻辑 | 正式评测不允许修改协议、断言和评分逻辑 |
| `ghrc_eval_orchestrator.py` | host | 批量拉取镜像、调用评测脚本、回写评测结果 | 评测平台侧维护，选手通常不需要修改 |

赛事交付中，选手应通过以下方式接入策略：

- 标准 LeRobot checkpoint：修改 `eval_config/eval_infer.yaml` 中的 `policy_type`、`policy_path` 或 `task_policy_paths`。
- 非标准策略或外部项目：新增自定义 `PolicyAdapter`，并在 `eval_config/eval_infer.yaml` 中配置 `adapter_class`。

---

## 3. 前置条件

| 项目 | 要求 |
| --- | --- |
| 镜像 | 已构建 `ghrc-eval-infer:latest` 和 `ghrc-eval-sim:latest` |
| 模型 | LeRobot checkpoint 或自定义 policy 所需权重已放入容器可访问路径 |
| GPU | `sim-eval` 容器需要可用 NVIDIA GPU；12GB 显存为最低运行要求 |
| 资源 | 仿真资产、任务配置和基础依赖已按赛事说明准备完成 |
| 网络 | 本机 8765 / 8766 端口未被其他进程占用 |

### 构建镜像

```bash
docker build -f docker/Dockerfile.eval_infer -t ghrc-eval-infer:latest .
docker build -f docker/Dockerfile.eval_sim   -t ghrc-eval-sim:latest .
```

---

## 4. 配置文件说明

| 配置文件 | 作用 | 常用修改项 |
| --- | --- | --- |
| `eval_config/eval_infer.yaml` | infer 容器配置 | `task`、`device`、`policy_type`、`policy_path`、`task_policy_paths`、`adapter_class`、`adapter_config` |
| `eval_config/eval_sim.yaml` | sim-eval 容器配置 | `task`、`device`、`num_episodes`、`auto_start`、`enable_assertion`、WebSocket 连接端口 |
| `eval_config/eval_orchestrator.yaml` | 自动化编排配置 | 飞书表格、镜像仓库、任务列表、超时时间、结果目录 |

任务名可通过 CLI `--task` 或脚本参数覆盖 YAML 中的 `task`。正式评测建议以脚本传入任务名，保证同一套配置可以复用于 `task1` 到 `task4`。

### 4.1 多任务 policy 路径

四个任务通常对应四个不同 checkpoint。不要依赖源码内置默认路径，建议在 `eval_config/eval_infer.yaml` 中显式配置：

```yaml
policy_type: act
policy_path: null
require_task_policy_paths: true
task_policy_paths:
  task1: ../challenge2026_baseline/task1/act/pretrained_model
  task2: ../challenge2026_baseline/task2/act/pretrained_model
  task3: ../challenge2026_baseline/task3/act/pretrained_model
  task4: ../challenge2026_baseline/task4/act/pretrained_model
```

路径解析规则：

- 绝对路径按原样使用。
- 相对路径按 YAML 文件所在目录解析，例如 `eval_config/eval_infer.yaml` 中的 `../xxx` 会解析到仓库根目录下的 `xxx`。
- `require_task_policy_paths: true` 时，当前任务缺少显式路径会直接报错，避免误用默认模型。

### 4.2 自定义 adapter 配置

自定义策略使用 `adapter_class` 接入：

```yaml
adapter_class: my_team_policy.ghrc_adapter:MyAdapter
adapter_config:
  action_dim: 20
policy_type: null
policy_path: /workspace/eval/my_team_policy/checkpoints/best.pt
```

详细接口、零动作示例和外部项目迁移方式见 [自定义策略接入指南](custom_policy.md)。

### 4.3 环境变量覆盖

常用运行时环境变量：

```bash
export INFER_IMAGE=ghrc-eval-infer:latest
export SIM_IMAGE=ghrc-eval-sim:latest
export INFER_CONFIG=eval_config/eval_infer.yaml
export SIM_CONFIG=eval_config/eval_sim.yaml
export INFER_READY_TIMEOUT=300
export HEADLESS=1
```

| 变量 | 说明 |
| --- | --- |
| `INFER_IMAGE` | infer 容器镜像 |
| `SIM_IMAGE` | sim-eval 容器镜像 |
| `INFER_CONFIG` | infer 配置文件路径 |
| `SIM_CONFIG` | sim-eval 配置文件路径 |
| `INFER_READY_TIMEOUT` | 等待 infer WebSocket 端口就绪的最长秒数 |
| `HEADLESS=1` | 无头模式，适合服务器和 CI |
| `HEADLESS=0` | 桌面调试模式，显示 Isaac Sim 窗口 |

---

## 5. 本地评测运行

### 5.1 运行

```bash
./run_eval.sh task4
./run_eval.sh all
```

运行流程：

1. 后台启动 infer 容器。
2. 等待 WebSocket 控制端口和数据端口就绪，默认 8765 / 8766。
3. 前台启动 sim-eval 容器。
4. sim-eval 连接 infer，逐步发送 observation 并接收 action。
5. episode 结束后写入评测结果和日志。

### 5.2 输出目录

| 输出 | 路径 |
| --- | --- |
| infer 关键日志 | `/tmp/eval_infer_{task}.log` |
| sim-eval 结果 | `logs/sim_eval_container/` |
| 自动化编排缓存 | `logs/eval_cache.csv` |

---

## 6. 自定义策略接入

评测系统只要求策略最终实现统一的推理接口：输入当前 observation，输出一维 action。推荐按复杂度选择接入方式：

| 接入方式 | 适用情况 | 文档 |
| --- | --- | --- |
| LeRobot 默认 adapter | checkpoint 符合 LeRobot `from_pretrained` 和 `select_action` 接口 | [自定义策略接入指南](custom_policy.md) |
| 自定义 `PolicyAdapter` | 自定义 PyTorch、ONNX、TensorRT、RL、规划器或混合算法 | [自定义策略接入指南](custom_policy.md) |
| 外部项目迁移 | 策略来自另一个完整项目文件夹，需要保留原目录结构 | [外部算法项目迁移示例](external_algorithm_migration.md) |

仓库提供两类最小验证示例：

| 示例 | 用途 |
| --- | --- |
| `eval_config/eval_infer_zero_action.yaml` | 输出全 0 action，验证 infer/sim 通信和 action 解码链路 |
| `eval_config/eval_infer_external_random.yaml` | 使用外部项目目录输出随机 action，验证项目 import、adapter 加载和迁移结构 |

注意：`curl http://localhost:8765/` 返回 `426 Upgrade Required` 表示该端口是 WebSocket 服务，不是普通 HTTP 页面。这通常说明 infer 服务已在监听，需使用 sim-eval 或 WebSocket 客户端连接。

---

## 7. 故障排查

| 问题 | 检查项 |
| --- | --- |
| infer 启动后无响应 | 查看 `docker logs eval_infer_{task}` 和 `/tmp/eval_infer_{task}.log` |
| `curl` 返回 426 | 正常现象；8765 是 WebSocket 控制端口，不是浏览器 HTTP 服务 |
| Isaac Sim 已打开但机器人不动 | 检查 `eval_config/eval_sim.yaml` 中 `auto_start` 是否为 `true`；为 `false` 时会等待键盘 Enter |
| sim-eval 连接失败 | 确认 infer 容器仍在运行，8765 / 8766 端口映射一致 |
| policy 路径错误 | 检查 `policy_path` 或 `task_policy_paths` 是否为容器内可访问路径 |
| `adapter_class` 导入失败 | 确认模块路径可 import，项目目录在 `PYTHONPATH` 中，并且每级包目录包含 `__init__.py` |
| action 维度异常 | 检查 `predict()` 是否返回一维 `torch.Tensor`、`np.ndarray` 或 `list[float]` |
| GPU 显存不足 | 12GB 显存为最低要求，可减少 `num_episodes` 或使用更高显存 GPU |

---

## 8. 自动化编排

编排系统从飞书多维表格读取选手提交的镜像，自动拉取镜像、运行评测、收集结果并回写分数。

### 8.1 前置条件

| 项目 | 要求 |
| --- | --- |
| 飞书应用 | 已创建企业自建应用，并获得 App ID / App Secret |
| 表格权限 | 应用已添加到目标多维表格 |
| 镜像仓库 | 评测机具备 Docker Hub、ACR 或企业私有仓库访问权限 |
| 本地环境 | 已配置 Docker、NVIDIA Container Toolkit 和评测镜像 |

### 8.2 飞书配置

1. 打开 [飞书开发者后台](https://open.feishu.cn)，创建企业自建应用，记录 **App ID** 和 **App Secret**。
2. 在应用详情的权限管理中勾选权限，并发布新版本。

| 权限 | 用途 |
| --- | --- |
| `bitable:app` | 读取多维表格 |
| `base:record:update` | 写入评测状态和分数 |

3. 打开飞书表格，选择右上角菜单，添加文档应用。
4. 从表格 URL 获取 `BITABLE_APP_TOKEN` 和 `TABLE_ID`。

```text
https://xxx.feishu.cn/base/BITABLE_APP_TOKEN?table=TABLE_ID
```

推荐表格列结构：

| 列名 | 类型 | 说明 |
| --- | --- | --- |
| 队伍编号 | 文本 | 唯一标识 |
| 队伍名称 | 文本 | 展示名称 |
| 镜像名称 | 链接或文本 | 选手 Docker 镜像地址 |
| 评审状态 | 文本 | 初始值为 `评测中` |
| 得分 | 文本 | 系统自动写入 |

### 8.3 编排器配置

`eval_config/eval_orchestrator.yaml` 示例：

```yaml
feishu:
  bitable_app_token: "xxx"
  table_id: "xxx"
registry:
  server: "your-registry.example.com"
eval:
  sim_image: "ghrc-eval-sim:latest"
  tasks: ["task4", "task1", "task2", "task3"]
  single_task_timeout: 7200
  results_dir: "logs/sim_eval_container"
```

### 8.4 凭据配置

```bash
cp .env.example .env
```

在 `.env` 中填写：

```bash
FEISHU_APP_ID=cli_xxxxxxxx
FEISHU_APP_SECRET=xxxxxxxx
DOCKER_USERNAME=xxxx
DOCKER_PASSWORD=xxxx
```

### 8.5 手动运行

```bash
source .env
export FEISHU_APP_ID FEISHU_APP_SECRET DOCKER_USERNAME DOCKER_PASSWORD
PYTHONPATH=src python -m lerobot.scripts.ghrc_eval_orchestrator
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--keep-images` | 评测后保留选手镜像 |
| `--skip-docker` | 跳过镜像拉取，适合本地调试 |
| `--mock-eval` | 生成模拟分数，适合联调飞书回写 |
| `--mock` | 完整模拟模式，不访问飞书和 Docker |
| `-v` | 输出详细日志 |

### 8.6 定时运行

```bash
crontab -e
```

```cron
*/30 * * * * source /path/to/.env && cd /path/to/project && export FEISHU_APP_ID FEISHU_APP_SECRET DOCKER_USERNAME DOCKER_PASSWORD && PYTHONPATH=src python -m lerobot.scripts.ghrc_eval_orchestrator >> /var/log/eval_orchestrator.log 2>&1
```

生产环境建议使用 `flock` 或调度平台锁机制，避免多个评测任务并发争抢 GPU。

### 8.7 编排状态

| 状态 | 含义 |
| --- | --- |
| `评测中` | 待评测，编排器会拾取该记录 |
| `评测完成` | 评测成功，分数已回写 |
| `评测失败` | 评测运行异常或超时 |
| `镜像异常` | 镜像拉取、登录或格式异常 |

已评测记录会自动跳过。需要重新评测时，将状态手动改回 `评测中`。

### 8.8 编排器故障排查

| 问题 | 检查项 |
| --- | --- |
| 飞书读写失败 | 权限是否开通并发布，应用是否已添加到表格 |
| Docker 登录失败 | `.env` 凭据是否正确，registry 地址是否匹配 |
| 镜像拉取超时 | 首次拉取大镜像可能需要 5 到 10 分钟，需确认网络和仓库限速 |
| 定时任务未执行 | 检查 `crontab -l` 和 `/var/log/eval_orchestrator.log` |
| 结果未回写 | 检查 `logs/sim_eval_container/` 是否生成 summary JSON |
