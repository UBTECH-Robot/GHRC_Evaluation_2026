import logging
from datetime import datetime
from typing import Any
import numpy as np
import torch
from .common import _make_json_safe

from pathlib import Path
import json

logger = logging.getLogger(__name__)


class EpisodeResult:
    """单次 episode 的推理结果。"""

    def __init__(
        self,
        episode_id: int,
        status: str = "unknown",
        steps: int = 0,
        score: int = 0,
        error_msg: str | None = None,
        assertion_result: str = "",
        metrics: dict | None = None,
    ):
        self.episode_id = episode_id
        self.status = status
        self.steps = steps
        self.score = int(score)
        self.error_msg = error_msg
        self.assertion_result = assertion_result
        self.metrics = metrics or {}
        self.start_time = datetime.now().isoformat()
        self.end_time = ""
        self.duration_seconds = 0.0

    def finalize(self) -> None:
        self.end_time = datetime.now().isoformat()
        try:
            self.duration_seconds = (
                datetime.fromisoformat(self.end_time) -
                datetime.fromisoformat(self.start_time)
            ).total_seconds()
        except Exception:
            pass

    def to_dict(self) -> dict:
        return _make_json_safe({
            "episode_id": self.episode_id,
            "status": self.status,
            "steps": self.steps,
            "score": self.score,
            "error_msg": self.error_msg,
            "assertion_result": self.assertion_result,
            "metrics": self.metrics,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
        })


class InferenceSummary:
    """多 episode 批量推理的汇总统计。"""

    def __init__(
        self,
        task_name: str,
        policy_type: str,
        policy_path: str,
        num_episodes: int,
        max_steps: int,
        task_text: str = "",
    ):
        self.task_name = task_name
        self.policy_type = policy_type
        self.policy_path = policy_path
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.task_text = task_text
        self.timestamp = datetime.now().isoformat()
        self.success_count = 0
        self.failed_count = 0
        self.error_count = 0
        self.total_steps = 0
        self.total_duration_seconds = 0.0
        self.details: list[dict] = []

    def add_result(self, result: EpisodeResult) -> None:
        if result.status == "success":
            self.success_count += 1
        elif result.status == "failed":
            self.failed_count += 1
        else:
            self.error_count += 1
        self.total_steps += result.steps
        self.total_duration_seconds += result.duration_seconds
        self.details.append(result.to_dict())

    @property
    def success_rate(self) -> float:
        return self.success_count / self.num_episodes if self.num_episodes > 0 else 0.0

    def to_dict(self) -> dict:
        episode_scores_with_reason: list[dict[str, Any]] = []
        for item in self.details:
            episode_scores_with_reason.append({
                "episode_id": int(item.get("episode_id", -1)),
                "score": int(item.get("score", 0)),
                "reason": str(item.get("assertion_result", "")),
                "status": str(item.get("status", "unknown")),
            })

        return _make_json_safe({
            "timestamp": self.timestamp,
            "task_name": self.task_name,
            "policy_type": self.policy_type,
            "policy_path": self.policy_path,
            "task_text": self.task_text,
            "num_episodes": self.num_episodes,
            "max_steps": self.max_steps,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "error_count": self.error_count,
            "total_steps": self.total_steps,
            "total_duration_seconds": self.total_duration_seconds,
            "success_rate": self.success_rate,
            "details": self.details,
            "episode_scores_with_reason": episode_scores_with_reason,
        })





def save_episode(log_dir: Path, result: EpisodeResult) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"episode_{result.episode_id:04d}.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)


def save_summary(log_dir: Path, summary: InferenceSummary) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = summary.to_dict()
    summary_file = log_dir / f"summary_{ts}.json"
    with open(summary_file, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"汇总结果已保存至 {summary_file}")


def print_summary(summary: InferenceSummary) -> None:
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  批量推理汇总  [{summary.task_name.upper()} / {summary.policy_type.upper()}]")
    print(sep)
    print(f"  策略路径  : {summary.policy_path}")
    if summary.task_text:
        print(f"  任务描述  : {summary.task_text}")
    print(f"  总 episodes: {summary.num_episodes}  |  最大步数：{summary.max_steps}")
    print(f"  成功：{summary.success_count}  失败：{summary.failed_count}  异常：{summary.error_count}")
    print(f"  成功率    : {summary.success_rate * 100:.2f}%")
    if summary.num_episodes > 0:
        print(f"  平均步数  : {summary.total_steps / summary.num_episodes:.1f}")
        print(f"  平均耗时  : {summary.total_duration_seconds / summary.num_episodes:.2f} s")
    print(sep)
