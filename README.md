# Global Humanoid Robot Challenge 2026 Evaluation System

Official technical documentation for the Global Humanoid Robot Challenge 2026 (GHRC 2026) evaluation system. The evaluation system adopts a dual-container isolation architecture: the `infer` container loads participant policies and provides action inference services via WebSocket, while the `sim-eval` container launches Isaac Sim, executes tasks, collects results and generates scoring logs. Observation and action data encoded with msgpack + lz4 are transmitted between the two containers via WebSocket.

## Project Overview

This document is intended for all participating teams of the **Global Humanoid Robot Challenge 2026 (GHRC 2026)**, fully covering the full workflow of evaluation system deployment & operation, custom policy integration, external project migration and work submission. Standardized and automated evaluation is implemented based on the dual-container isolation architecture.

## Resource Description

This project requires simulation environment and robot assets hosted on Hugging Face. **Please complete the download before first use**:

| Resource Type | Local Directory | Remote Address |
| --- | --- | --- |
| 🤖 Simulation Environment & Robot Assets | `assets/` (Git submodule) | [UBTECH-Robotics/challenge2026_assets](https://huggingface.co/UBTECH-Robotics/challenge2026_assets) |

### Configuration Information

| | Minimum Requirements | Recommended Configuration | Ideal Configuration |
| --- | --- | --- | --- |
| **Operating System** | Ubuntu 22.04 / 24.04; Windows 10 / 11 | Ubuntu 22.04 / 24.04; Windows 10 / 11 | Ubuntu 22.04 / 24.04; Windows 10 / 11 |
| **CPU** | Intel Core i7 (7th Generation); AMD Ryzen 5 | Intel Core i7 (9th Generation); AMD Ryzen 7 | Intel Core i9, X-series or higher; AMD Ryzen 9, Threadripper or higher |
| **Core Count** | 4 | 8 | 16 |
| **RAM** | 32GB | 64GB | 64GB |
| **Storage** | 50GB SSD | 500GB SSD | 1TB NVMe SSD |
| **GPU** | GeForce RTX 4080 | GeForce RTX 5080 | RTX PRO 6000 Blackwell |
| **VRAM** | 16GB | 16GB | 48GB |
| **Driver** | Linux: 580.65.06; Windows: 580.88 | Linux: 580.65.06; Windows: 580.88 | Linux: 580.65.06; Windows: 580.88 |

> It is recommended to keep the configuration consistent with the baseline requirements.

### Tool Requirements

| Tool | Version | Notes |
| --- | --- | --- |
| `CUDA` | 12.8 | [Official Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) |
| `Docker` | latest | [Official Guide](https://docs.docker.com/engine/install/ubuntu/) |
| `NVIDIA Container Toolkit` | latest | [Official Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| `Hugging Face` | latest | `pip install huggingface-hub`; verify installation with `huggingface-cli --help` |
| `Git` | latest | `sudo apt update`; `sudo apt install git -y`; verify version with `git --version` |
| `Miniconda` | latest | [Official Guide](https://www.anaconda.com/docs/getting-started/miniconda/install/overview) (Optional) |

## Technical Documentation Index

The evaluation system is divided into four parts: **it is recommended to read them in order**; you can also jump directly to the required section according to your actual progress.

| No. | Document | Description |
|---|------|------|
| 1 | [Evaluation User Guide](https://docs.ubtrobot.com/GHRC2026_EvalDocuments/en/docs/1/) | Explains the operation mode, configuration boundaries and custom policy integration entry of the evaluation system. |
| 2 | [Custom Policy Integration Guide](https://docs.ubtrobot.com/GHRC2026_EvalDocuments/en/docs/2) | Explains how to integrate a custom policy into the GHRC infer container. | GHRC infer container. |
| 3 | [External Algorithm Migration Guide](https://docs.ubtrobot.com/GHRC2026_EvalDocuments/en/docs/3) | Explains file placement, dependency declaration, integration with evaluation via `PolicyAdapter`, and pipeline verification with random action examples when migrating non-LeRobot projects to the GHRC evaluation repository. |
| 4 | [Participant Submission Specification Guide](https://docs.ubtrobot.com/GHRC2026_EvalDocuments/en/docs/4) | Explains the project structure, packaging operations and common issues for participant submissions. |
