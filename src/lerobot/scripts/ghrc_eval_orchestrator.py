#!/usr/bin/env python3
"""GHRC 2026 评测编排器 — 自动化评测流程入口。

负责：
1. 从飞书多维表格（或本地 JSON）读取选手提交的镜像信息
2. 从 Docker registry 拉取选手镜像（支持 mock 跳过）
3. 调用 run_eval.sh 执行双容器评测（支持 mock 生成假结果）
4. 收集评测结果并回写飞书多维表格

使用示例:

    # 生产模式（需要飞书 + Docker registry 凭据）
    python -m lerobot.scripts.ghrc_eval_orchestrator

    # 本地测试模式（无需外部依赖）
    python -m lerobot.scripts.ghrc_eval_orchestrator --mock --mock-competitors mock_competitors.json

    # 仅校验配置
    python -m lerobot.scripts.ghrc_eval_orchestrator --validate

环境变量（敏感信息，不入配置文件）:
    FEISHU_APP_ID         飞书应用 App ID
    FEISHU_APP_SECRET     飞书应用 App Secret
    DOCKER_USERNAME       Docker registry 用户名
    DOCKER_PASSWORD       Docker registry 密码
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from src.lerobot.sim_eval.const import TASK_CHOICES
from src.lerobot.sim_eval.container_config import load_container_config, require_config_keys
from src.lerobot.sim_eval.registry import (
    DockerRegistryClient,
    check_disk_space,
    retry,
    validate_image_name,
)
from src.lerobot.sim_eval.spreadsheet import (
    STATUS_DONE,
    STATUS_EVALUATING,
    STATUS_FAILED,
    STATUS_IMAGE_ERROR,
    CompetitorInfo,
    EvalScore,
    FeishuBitableClient,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval_orchestrator")

_EVAL_SCRIPT = "run_eval.sh"
_MOCK_TASKS = list(TASK_CHOICES)
_DIGEST_STATE_FILE = "logs/sim_eval_container/processed_digests.json"
_PER_COMPETITOR_TIMEOUT = 10800  # 单选手总超时：3 小时


# ======================================================================
# 本地 JSON 选手数据源（mock 模式使用）
# ======================================================================

class LocalCompetitorSource:
    """从本地 JSON 文件读取选手列表，用于离线测试。"""

    def __init__(self, json_path: str) -> None:
        self._path = Path(json_path)

    def fetch_competitors(self, status_filter: list[str] | None = None) -> list[CompetitorInfo]:
        if not self._path.exists():
            raise FileNotFoundError(f"选手 JSON 文件不存在：{self._path}")

        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data if isinstance(data, list) else data.get("competitors", [])
        competitors: list[CompetitorInfo] = []
        default_filter = status_filter or [STATUS_EVALUATING, ""]

        for i, record in enumerate(records):
            if isinstance(record, str):
                record = {"name": f"competitor_{i}", "image_name": record}
            status = record.get("status", STATUS_EVALUATING)
            if status not in default_filter:
                continue

            competitors.append(
                CompetitorInfo(
                    record_id=record.get("record_id", f"local_{i}"),
                    team_id=record.get("team_id", f"local_{i}"),
                    name=record.get("name", f"unknown_{i}"),
                    email=record.get("email", ""),
                    image_name=record.get("image_name", ""),
                    status=status,
                )
            )
        return competitors


# ======================================================================
# 本地缓存（飞书故障时的保底方案，CSV 格式可用 Excel 打开）
# ======================================================================

import csv

_CSV_COLUMNS = ["队伍编号", "队伍名称", "队长邮箱", "镜像名称", "评审状态", "得分", "评测详情", "评测时间"]


class LocalCache:
    """本地 CSV 缓存，可直接用 Excel/WPS 打开查看。

    读：飞书失败时回退到缓存
    写：每次成功都更新缓存，确保始终有一份本地副本。
    """

    def __init__(self, cache_path: Path) -> None:
        self._path = cache_path

    def save_competitors(self, competitors: list[CompetitorInfo]) -> None:
        """保存选手基本信息，保留已有的评测结果。"""
        # 读出现有数据中的分数和详情
        existing_scores: dict[str, list[str]] = {}
        if self._path.exists():
            for row in self._read_rows():
                if len(row) >= 7 and row[0]:
                    existing_scores[row[0]] = [row[4], row[5], row[6], row[7]]  # 状态,得分,详情,时间

        rows = []
        for c in competitors:
            prev = existing_scores.get(c.team_id, ["", "", "", ""])
            rows.append([c.team_id, c.name, c.email, c.image_name,
                         prev[0] or c.status, prev[1] or "", prev[2] or "", prev[3] or ""])
        self._write_rows(rows)
        logger.info("本地缓存已更新 (%s 条选手)", len(competitors))

    def load_competitors(self, status_filter: list[str] | None = None) -> list[CompetitorInfo]:
        default_filter = status_filter or [STATUS_EVALUATING, ""]
        if not self._path.exists():
            return []
        rows = self._read_rows()
        competitors: list[CompetitorInfo] = []
        for i, row in enumerate(rows):
            if len(row) < 5:
                continue
            status = row[4] if len(row) > 4 else ""
            if status not in default_filter:
                continue
            competitors.append(CompetitorInfo(
                record_id=f"cache_{i}",
                team_id=row[0] if len(row) > 0 else "",
                name=row[1] if len(row) > 1 else "",
                email=row[2] if len(row) > 2 else "",
                image_name=row[3] if len(row) > 3 else "",
                status=status,
            ))
        return competitors

    def save_result(self, competitor: CompetitorInfo, score: EvalScore) -> None:
        """更新单个选手的分数列，按 team_id 匹配。"""
        rows = self._read_rows() if self._path.exists() else []
        found = False
        for row in rows:
            if len(row) >= 1 and row[0] == competitor.team_id:
                # 更新已有行: 状态、得分、详情、时间
                while len(row) < 8:
                    row.append("")
                row[4] = score.status
                row[5] = str(score.score)
                row[6] = score.detail
                row[7] = time.strftime("%Y-%m-%d %H:%M:%S")
                found = True
                break
        if not found:
            rows.append([competitor.team_id, competitor.name, competitor.email,
                         competitor.image_name, score.status, str(score.score),
                         score.detail, time.strftime("%Y-%m-%d %H:%M:%S")])
        self._write_rows(rows)

    def _read_rows(self) -> list[list[str]]:
        """读取所有数据行（跳过表头和空行）。"""
        with open(self._path, "r", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        # 跳过表头行，跳过空行
        return [r for r in rows[1:] if any(c.strip() for c in r)]

    def _write_rows(self, rows: list[list[str]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_CSV_COLUMNS)
            writer.writerows(rows)


# ======================================================================
# Mock 评测（生成假 summary 文件）
# ======================================================================

def _run_mock_eval(results_dir: Path) -> None:
    """生成模拟的评测 summary JSON 文件，用于流程验证。"""
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp_base = time.time()
    for idx, task in enumerate(_MOCK_TASKS):
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp_base + idx))
        summary_file = results_dir / f"summary_{ts}.json"

        success_rate = round(random.uniform(0.3, 0.95), 3)
        num_episodes = random.randint(3, 10)
        success_count = int(num_episodes * success_rate)
        failed_count = num_episodes - success_count

        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(timestamp_base + idx)),
            "task_name": task,
            "policy_type": "act",
            "policy_path": f"mock://competitor/{task}",
            "task_text": f"Mock evaluation for {task}",
            "num_episodes": num_episodes,
            "max_steps": 500,
            "success_count": success_count,
            "failed_count": failed_count,
            "error_count": 0,
            "total_steps": num_episodes * random.randint(100, 400),
            "total_duration_seconds": num_episodes * random.uniform(30, 120),
            "success_rate": success_rate,
            "details": [],
            "episode_scores_with_reason": [
                {
                    "episode_id": ep,
                    "score": random.randint(40, 100),
                    "reason": "mock_success" if ep < success_count else "mock_terminal",
                    "status": "success" if ep < success_count else "failed",
                }
                for ep in range(num_episodes)
            ],
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Mock summary: %s", summary_file.name)


# ======================================================================
# 编排器主类
# ======================================================================

class EvalOrchestrator:
    """评测编排器。

    串联：飞书表格读取 → 镜像拉取 → 评测执行 → 结果回写。
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mock_mode: bool = bool(getattr(args, "mock_mode", False))
        self.skip_docker: bool = self.mock_mode or bool(getattr(args, "skip_docker", False))
        self.mock_eval: bool = self.mock_mode or bool(getattr(args, "mock_eval", False))
        self.keep_images: bool = bool(getattr(args, "keep_images", False))
        self.mock_competitors_file: str = str(getattr(args, "mock_competitors_file", "") or "")

        self.project_dir = Path(args.project_dir).resolve()
        self.results_dir = Path(args.eval_results_dir)  # parse_args 已解析为绝对路径
        self.eval_script = self.project_dir / _EVAL_SCRIPT
        self.sim_image = str(args.eval_sim_image)
        self.tasks: list[str] = list(args.eval_tasks)
        self.single_task_timeout = int(args.eval_single_task_timeout)
        self.total_timeout = self.single_task_timeout * len(self.tasks) + 600

        self.feishu: FeishuBitableClient | None = None
        self.registry: DockerRegistryClient | None = None
        self.local_source: LocalCompetitorSource | None = None

        # 优雅关闭
        self._shutdown_flag = False
        self._current_competitor: CompetitorInfo | None = None

        # 本地缓存（飞书保底方案）
        self._cache = LocalCache(self.project_dir / "logs" / "eval_cache.csv")

        # 镜像摘要去重：{digest: team_name} 防止重复评测
        self._digest_state_path = self.project_dir / _DIGEST_STATE_FILE
        self._processed_digests: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        _init_logging(self.args)

        if self.mock_mode or self.mock_competitors_file:
            self._init_local_source()
        else:
            self._init_feishu()

        if not self.skip_docker:
            self._init_registry()

        self._load_digest_state()
        self._validate_environment()
        _print_banner(self)

    def _load_digest_state(self) -> None:
        """加载已处理镜像摘要（防重复评测）。"""
        if self._digest_state_path.exists():
            try:
                with open(self._digest_state_path, "r", encoding="utf-8") as f:
                    self._processed_digests = json.load(f)
                logger.info("已加载 %s 条已处理镜像摘要", len(self._processed_digests))
            except (json.JSONDecodeError, OSError):
                self._processed_digests = {}

    def _save_digest_state(self) -> None:
        """持久化已处理镜像摘要。"""
        self._digest_state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._digest_state_path, "w", encoding="utf-8") as f:
            json.dump(self._processed_digests, f, indent=2, ensure_ascii=False)

    def _init_feishu(self) -> None:
        app_id = os.getenv("FEISHU_APP_ID", "")
        app_secret = os.getenv("FEISHU_APP_SECRET", "")
        if not app_id or not app_secret:
            raise RuntimeError("缺少飞书凭据，请设置 FEISHU_APP_ID / FEISHU_APP_SECRET")
        self.feishu = FeishuBitableClient(
            app_id=app_id,
            app_secret=app_secret,
            bitable_app_token=str(self.args.feishu_bitable_app_token),
            table_id=str(self.args.feishu_table_id),
        )
        logger.info("飞书客户端初始化完成")

    def _init_local_source(self) -> None:
        file_path = self.mock_competitors_file or "mock_competitors.json"
        self.local_source = LocalCompetitorSource(file_path)
        logger.info("本地选手数据源：%s", file_path)

    def _init_registry(self) -> None:
        username = os.getenv("DOCKER_USERNAME", "")
        password = os.getenv("DOCKER_PASSWORD", "")
        if not username or not password:
            raise RuntimeError("缺少 Docker 凭据，请设置 DOCKER_USERNAME / DOCKER_PASSWORD")
        self.registry = DockerRegistryClient(
            registry=str(getattr(self.args, "registry_server", "docker.io")),
            username=username,
            password=password,
        )
        logger.info("Docker registry 客户端初始化完成")

    def _validate_environment(self) -> None:
        if not self.mock_eval and not self.eval_script.exists():
            raise FileNotFoundError(f"评测脚本不存在：{self.eval_script}")
        if not self.skip_docker and not self.mock_eval:
            if shutil.which("docker") is None:
                raise RuntimeError("未找到 docker")

    def _setup_signal_handlers(self) -> None:
        def handler(signum: int, _frame: Any) -> None:
            name = signal.Signals(signum).name
            logger.warning("收到 %s 信号，准备优雅退出...", name)
            self._shutdown_flag = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, handler)

    def _health_check(self) -> bool:
        """启动前健康检查，失败时不应继续。"""
        ok = True

        # 1. 磁盘空间
        if not self.mock_mode and not self.skip_docker:
            if not check_disk_space():
                ok = False

        # 2. Docker daemon
        if not self.skip_docker and not self.mock_eval:
            try:
                subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=True)
                logger.info("Docker daemon: OK")
            except Exception as exc:
                logger.error("Docker daemon 不可用: %s", exc)
                ok = False

        # 3. 飞书连通性
        if not self.mock_mode and self.feishu is not None:
            try:
                competitors = self.feishu.fetch_competitors()
                logger.info("飞书连通性: OK (%s 条待评测)", len(competitors))
            except Exception as exc:
                logger.error("飞书连通性检查失败: %s", exc)
                ok = False

        if not ok:
            logger.error("健康检查未通过，终止")
        return ok

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._setup_signal_handlers()

        logger.info("=" * 60)
        logger.info("评测编排器启动%s", " [MOCK 模式]" if self.mock_mode else "")
        logger.info("=" * 60)

        if not self._health_check():
            return

        competitors = self._fetch_competitors()
        if not competitors:
            logger.info("无待评测选手，退出")
            return

        logger.info("共 %s 位选手待评测", len(competitors))

        # Registry 登录
        if self.registry is not None:
            try:
                self.registry.login()
            except Exception as exc:
                logger.error("Registry 登录失败，终止：%s", exc)
                return

        success_count = 0
        for idx, competitor in enumerate(competitors):
            if self._shutdown_flag:
                logger.warning("收到退出信号，跳过剩余 %s 位选手", len(competitors) - idx)
                break

            self._current_competitor = competitor
            logger.info("=" * 40)
            logger.info(
                "[%s/%s] %s (%s) | 镜像：%s",
                idx + 1, len(competitors),
                competitor.name, competitor.team_id, competitor.image_name,
            )
            logger.info("=" * 40)
            try:
                self._evaluate_one_with_timeout(competitor)
                success_count += 1
            except Exception as exc:
                logger.error("选手 %s 异常：%s", competitor.name, exc)
                logger.error(traceback.format_exc())
                self._mark_failed_safe(competitor, str(exc))

        self._current_competitor = None
        logger.info("=" * 60)
        logger.info("本轮完成：%s/%s 成功", success_count, len(competitors))
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 单选手评测
    # ------------------------------------------------------------------

    def _evaluate_one_with_timeout(self, competitor: CompetitorInfo) -> None:
        """带总超时的单选手评测，超时自动中断。"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._evaluate_one, competitor)
            try:
                future.result(timeout=_PER_COMPETITOR_TIMEOUT)
            except concurrent.futures.TimeoutError:
                logger.error("选手 %s 评测超时 (%ss)，强制终止", competitor.name, _PER_COMPETITOR_TIMEOUT)
                self._shutdown_flag = True  # 防止后续选手继续跑
                self._mark_failed_safe(competitor, f"超出单选手总时间限制（{_PER_COMPETITOR_TIMEOUT}s）")
                raise RuntimeError(f"评测超时: {competitor.name}")

    def _evaluate_one(self, competitor: CompetitorInfo) -> None:
        image_name = competitor.image_name.strip()
        if not validate_image_name(image_name):
            raise ValueError(f"镜像名格式不合法：{image_name}")

        # 拉取镜像
        if not self.skip_docker and self.registry is not None:
            try:
                self.registry.pull_image(image_name)
            except Exception as exc:
                logger.error("镜像拉取失败：%s", exc)
                self._update_score_safe(
                    competitor,
                    EvalScore(status=STATUS_IMAGE_ERROR, detail=f"镜像拉取失败: {exc}"),
                )
                return
        else:
            logger.info("[mock] 跳过镜像拉取：%s", image_name)

        # 记录镜像摘要
        if self.registry is not None:
            digest = self.registry.get_image_digest(image_name)
            if digest:
                logger.info("镜像摘要：%s", digest)

        # 执行评测
        score = self._run_eval(image_name)

        # 回写分数
        self._update_score_safe(competitor, score)

        # 清理本地镜像
        if self.keep_images:
            logger.info("保留镜像：%s", image_name)
        elif self.registry is not None:
            try:
                self.registry.remove_image(image_name)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 评测执行与结果收集
    # ------------------------------------------------------------------

    def _run_eval(self, infer_image: str) -> EvalScore:
        _clear_results_dir(self.results_dir)

        if self.mock_eval:
            logger.info("[mock] 生成模拟评测结果...")
            _run_mock_eval(self.results_dir)
        else:
            self._run_eval_subprocess(infer_image)

        return self._collect_scores()

    def _run_eval_subprocess(self, infer_image: str) -> None:
        env = os.environ.copy()
        env["INFER_IMAGE"] = infer_image
        env["SIM_IMAGE"] = self.sim_image
        env["HEADLESS"] = "1"

        logger.info("启动评测子进程...")
        start_time = time.monotonic()

        try:
            proc = subprocess.run(
                ["bash", str(self.eval_script), "all"],
                cwd=str(self.project_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.total_timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error("评测超时 (%s 秒)", self.total_timeout)
            raise RuntimeError(f"评测超时（{self.total_timeout}秒）")

        elapsed = time.monotonic() - start_time
        logger.info("评测结束，exit=%s，耗时 %.0f 秒", proc.returncode, elapsed)

        if proc.stdout:
            for line in proc.stdout.strip().splitlines()[-20:]:
                logger.info("[eval] %s", line)
        if proc.returncode != 0 and proc.stderr:
            for line in proc.stderr.strip().splitlines()[-10:]:
                logger.warning("[eval-err] %s", line)

    def _collect_scores(self) -> EvalScore:
        """从 summary JSON 中提取分数。

        汇总逻辑：取各任务成功率的加权平均作为最终得分（0-100）。
        同时记录各任务详情到 detail 字段。
        """
        summary_files = sorted(self.results_dir.glob("summary_*.json"))
        if not summary_files:
            return EvalScore(status=STATUS_FAILED, detail="未找到评测结果文件")

        per_task: dict[str, float] = {}
        details_parts: list[str] = []

        for filepath in summary_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("无法读取 summary：%s — %s", filepath.name, exc)
                continue

            task_name = str(data.get("task_name", "")).lower()
            success_rate = float(data.get("success_rate", 0))
            success_count = int(data.get("success_count", 0))
            num_episodes = int(data.get("num_episodes", 1))

            per_task[task_name] = success_rate * 100.0
            details_parts.append(
                f"{task_name}: {success_count}/{num_episodes} ({success_rate*100:.1f}%)"
            )
            logger.info("  %s: rate=%.1f%% (%s/%s)", task_name, success_rate * 100, success_count, num_episodes)

        if not per_task:
            return EvalScore(status=STATUS_FAILED, detail="所有 summary 文件解析失败")

        # 总分 = 各任务加权平均
        overall = round(sum(per_task.values()) / len(per_task), 1)

        return EvalScore(
            score=overall,
            status=STATUS_DONE,
            detail=" | ".join(details_parts),
        )

    # ------------------------------------------------------------------
    # 选手数据获取
    # ------------------------------------------------------------------

    def _fetch_competitors(self) -> list[CompetitorInfo]:
        if self.local_source is not None:
            try:
                return self.local_source.fetch_competitors()
            except Exception as exc:
                logger.error("本地选手数据读取失败：%s", exc)
                return []
        return self._fetch_competitors_with_retry()

    def _fetch_competitors_with_retry(self) -> list[CompetitorInfo]:
        if self.feishu is None:
            return []
        for attempt in range(1, 3 + 1):  # _MAX_RETRIES = 3
            try:
                result = self.feishu.fetch_competitors()
                if result:
                    self._cache.save_competitors(result)
                return result
            except Exception as exc:
                logger.warning("飞书读取失败 (attempt %s/3): %s", attempt, exc)
                if attempt < 3:
                    time.sleep(min(10 * attempt, 30))

        # 飞书彻底失败 → 回退到本地缓存
        logger.error("飞书读取最终失败，回退到本地缓存")
        cached = self._cache.load_competitors()
        if cached:
            logger.warning("使用缓存数据：%s 条记录（可能不是最新）", len(cached))
            return cached
        logger.error("本地缓存也无数据，本轮无法评测")
        return []

    # ------------------------------------------------------------------
    # 安全的回写操作
    # ------------------------------------------------------------------

    @retry(max_attempts=3, base_delay=5.0)
    def _feishu_write(self, competitor: CompetitorInfo, score: EvalScore) -> None:
        """带重试的飞书回写。"""
        assert self.feishu is not None
        self.feishu.update_score(competitor, score)

    def _update_score_safe(self, competitor: CompetitorInfo, score: EvalScore) -> None:
        logger.info("结果: score=%.1f status=%s detail=%s", score.score, score.status, score.detail)

        # 始终写本地缓存（保底）
        self._cache.save_result(competitor, score)

        if self.mock_mode or self.feishu is None:
            logger.info("[mock] 跳过飞书回写")
            out = {
                "competitor": competitor.name,
                "team_id": competitor.team_id,
                "image_name": competitor.image_name,
                "score": score.score,
                "status": score.status,
                "detail": score.detail,
                "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            local_path = self.results_dir / f"result_{competitor.name}.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            logger.info("本地结果已保存 → %s", local_path)
            return
        try:
            self._feishu_write(competitor, score)
        except Exception as exc:
            logger.error("飞书回写最终失败: %s", exc)

    def _mark_failed_safe(self, competitor: CompetitorInfo, error_msg: str) -> None:
        self._update_score_safe(
            competitor, EvalScore(status=STATUS_FAILED, detail=error_msg)
        )


# ======================================================================
# 工具函数
# ======================================================================

def _init_logging(args: argparse.Namespace) -> None:
    level = logging.DEBUG if bool(getattr(args, "verbose", False)) else logging.INFO
    logging.getLogger().setLevel(level)
    for name in ("websockets", "websockets.client", "urllib3", "requests"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _print_banner(orch: EvalOrchestrator) -> None:
    logger.info("  Project:     %s", orch.project_dir)
    logger.info("  EvalScript:  %s", orch.eval_script)
    logger.info("  ResultsDir:  %s", orch.results_dir)
    logger.info("  SimImage:    %s", orch.sim_image)
    logger.info("  Tasks:       %s", orch.tasks)
    logger.info("  Timeout:     %ss/task (total %ss)", orch.single_task_timeout, orch.total_timeout)
    logger.info("  Mock:        %s", orch.mock_mode)
    logger.info("  SkipDocker:  %s", orch.skip_docker)
    logger.info("  MockEval:    %s", orch.mock_eval)


def _clear_results_dir(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("summary_*.json", "episode_*.json", "result_*.json"):
        for f in results_dir.glob(pattern):
            f.unlink()


# ======================================================================
# 命令行入口
# ======================================================================

def _flatten_nested_keys(
    args: argparse.Namespace,
    mapping: dict[str, list[str]],
) -> None:
    """将 YAML 嵌套键展平为 namespace 属性。

    YAML 中 feishu.bitable_app_token 在 namespace 中成了
    args.feishu (一个 dict)。此函数将其展平为 args.feishu_bitable_app_token。
    """
    for section, keys in mapping.items():
        section_dict = getattr(args, section, None)
        if not isinstance(section_dict, dict):
            continue
        for key in keys:
            if key in section_dict:
                setattr(args, f"{section}_{key}", section_dict[key])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GHRC 评测编排器")
    parser.add_argument("--config", type=str, default="eval_config/eval_orchestrator.yaml", help="配置文件路径")
    parser.add_argument("--project-dir", type=str, default=None, help="项目根目录（自动检测）")

    parser.add_argument("--mock", action="store_true", default=False, help="启用完整 mock 模式")
    parser.add_argument("--mock-competitors", type=str, default="", help="本地选手 JSON（mock 飞书）")
    parser.add_argument("--mock-eval", action="store_true", default=False, help="生成模拟评测结果")
    parser.add_argument("--skip-docker", action="store_true", default=False, help="跳过 Docker 镜像拉取")
    parser.add_argument("--keep-images", action="store_true", default=False, help="评测后保留本地镜像（不自动删除）")

    parser.add_argument("--validate", action="store_true", default=False, help="仅校验配置后退出")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="输出调试日志")

    cli_args = parser.parse_args()

    args = load_container_config(
        config_path=cli_args.config,
        defaults={
            "feishu_bitable_app_token": "",
            "feishu_table_id": "",
            "registry_server": "docker.io",
            "eval_sim_image": "ghrc-eval-sim:latest",
            "eval_tasks": ["task4", "task1", "task2", "task3"],
            "eval_single_task_timeout": 7200,
            "eval_results_dir": "logs/sim_eval_container",
            "schedule_mode": "once",
            "schedule_daily_time": "02:00",
        },
        path_fields=set(),
    )

    args.mock_mode = cli_args.mock
    args.mock_competitors_file = cli_args.mock_competitors
    args.mock_eval = cli_args.mock_eval or cli_args.mock
    args.skip_docker = cli_args.skip_docker or cli_args.mock
    args.validate = cli_args.validate
    args.verbose = cli_args.verbose

    # 将嵌套 YAML 配置展平到 namespace（feishu.bitable_app_token → feishu_bitable_app_token）
    _flatten_nested_keys(args, {
        "feishu": ["bitable_app_token", "table_id"],
        "registry": ["server"],
        "eval": ["sim_image", "tasks", "single_task_timeout", "results_dir"],
        "schedule": ["mode", "daily_time"],
    })

    if not cli_args.mock and not cli_args.mock_competitors:
        require_config_keys(args, ["feishu_bitable_app_token", "feishu_table_id"], "编排器配置")

    if cli_args.project_dir:
        args.project_dir = str(Path(cli_args.project_dir).resolve())
    else:
        args.project_dir = str(Path(__file__).resolve().parents[3])

    raw_results_dir = str(getattr(args, "eval_results_dir", "logs/sim_eval_container"))
    args.eval_results_dir = str(Path(args.project_dir) / raw_results_dir)

    return args


def main() -> None:
    args = parse_args()
    _init_logging(args)

    if args.validate:
        logger.info("配置校验通过")
        return

    orchestrator = EvalOrchestrator(args)
    try:
        orchestrator.initialize()
        orchestrator.run()
    except KeyboardInterrupt:
        logger.info("收到中断信号，退出")
    except Exception as exc:
        logger.error("编排器异常：%s", exc)
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
