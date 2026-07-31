"""Console formatting helpers for evaluation scores."""

from __future__ import annotations

from typing import Any


def format_task4_score_details(metrics: dict[str, Any]) -> str:
    """Format Task4 score components and explain the time-score status."""

    short_score = int(metrics.get("task4_short_edge_score", 0))
    long_score = int(metrics.get("task4_long_edge_score", 0))
    time_score = int(metrics.get("task4_time_score", 0))

    if not bool(metrics.get("is_success", False)):
        time_details = f"时间: {time_score}（任务成功后结算）"
    else:
        elapsed = float(metrics.get("task4_elapsed_seconds", 0.0))
        full_time = float(metrics.get("task4_time_full_time_seconds", 180.0))
        if elapsed <= full_time:
            time_details = (
                f"时间: {time_score}（{elapsed:.1f}s <= {full_time:.1f}s，时间达标）"
            )
        else:
            time_details = (
                f"时间: {time_score}（{elapsed:.1f}s > {full_time:.1f}s，超时扣分）"
            )

    return f"短边: {short_score} | 长边: {long_score} | {time_details}"
