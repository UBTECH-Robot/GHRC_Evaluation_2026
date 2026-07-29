# 全球人形机器人挑战赛 2026 评测系统

全球人形机器人挑战赛 2026（Global Humanoid Robot Challenge 2026）官方评测系统技术文档。评测系统采用双容器隔离架构：`infer` 容器负责加载选手策略并通过 WebSocket 提供动作推理服务，`sim-eval` 容器负责启动 Isaac Sim、执行任务、采集结果和生成评分日志。两个容器之间使用 WebSocket 传输 msgpack + lz4 编码的观测与动作数据。

## 项目概述


本文档面向 **全球人形机器人挑战赛 2026 (GHRC 2026)** 所有参赛队伍，完整覆盖评测系统部署运行、自定义策略接入、外部项目迁移与作品提交全流程，基于双容器隔离架构实现标准化自动化评测。


## 资源说明

本项目需要用到仿真环境与机器人资产，托管于 Hugging Face，首次使用前 **请先完成下载**：

| 资源类别 | 本地目录 | 远程地址 |
| --- | --- | --- |
| 🤖 仿真环境与机器人资产 | `assets/`（Git 子模块） | [UBTECH-Robotics/challenge2026_assets](https://huggingface.co/UBTECH-Robotics/challenge2026_assets) |


### 配置信息

| | 最低要求 | 推荐配置 | 理想配置 |
| --- | --- | --- | --- |
| **操作系统** | Ubuntu 22.04 / 24.04; Windows 10 / 11 | Ubuntu 22.04 / 24.04; Windows 10 / 11 | Ubuntu 22.04 / 24.04; Windows 10 / 11 |
| **CPU** | Intel Core i7 (7th Generation); AMD Ryzen 5 | Intel Core i7 (9th Generation); AMD Ryzen 7 | Intel Core i9, X-series or higher; AMD Ryzen 9, Threadripper or higher |
| **核心数** | 4 | 8 | 16 |
| **RAM** | 32GB | 64GB | 64GB |
| **存储空间** | 50GB SSD | 500GB SSD | 1TB NVMe SSD |
| **GPU** | GeForce RTX 4080 | GeForce RTX 5080 | RTX PRO 6000 Blackwell |
| **VRAM** | 16GB | 16GB | 48GB |
| **驱动** | Linux: 580.65.06; Windows: 580.88 | Linux: 580.65.06; Windows: 580.88 | Linux: 580.65.06; Windows: 580.88 |

> 建议配置与 baseline 基线要求一致

### 工具要求

| 工具 | 版本 | 备注 |
| --- | --- | --- |
| `CUDA` | 12.8 | [官方指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) |
| `Docker` | latest | [官方指南](https://docs.docker.com/engine/install/ubuntu/) |
| `NVIDIA Container Toolkit` | latest | [官方指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| `Hugging Face` | latest | `pip install huggingface-hub`; `huggingface-cli --help`（验证安装） |
| `Git` | latest | `sudo apt update`; `sudo apt install git -y`; `git --version`（验证版本） |
| `Miniconda` | latest | [官方指南](https://www.anaconda.com/docs/getting-started/miniconda/install/overview)（可选） |

## 技术文档索引

评测系统分为四个部分：**建议按顺序阅读**；也可根据实际进度直接跳转到所需阶段。

| # | 文档 | 说明 |
|---|------|------|
| 1 | [评测使用指南](https://docs.ubtrobot.com/GHRC2026_EvalDocuments/docs/1) | 说明评测系统运行方式、配置边界、自定义策略接入入口。 |
| 2 | [自定义策略接入指南](https://docs.ubtrobot.com/GHRC2026_EvalDocuments/docs/2) | 说明如何把自定义 policy 接入 GHRC infer 容器。 |
| 3 | [外部算法迁移指南](https://docs.ubtrobot.com/GHRC2026_EvalDocuments/docs/3) | 说明把非 LeRobot 项目迁移到 GHRC 评测仓库时，文件应如何放置、如何声明依赖、如何通过 `PolicyAdapter` 接入评测，以及如何用随机动作示例验证迁移链路。 |
| 4 | [选手提交作品规范指南](https://docs.ubtrobot.com/GHRC2026_EvalDocuments/docs/4) | 说明选手提交项目的结构、打包操作、常见问题。 |
