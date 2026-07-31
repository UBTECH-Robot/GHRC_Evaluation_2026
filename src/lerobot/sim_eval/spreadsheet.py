"""飞书多维表格（Bitable）客户端。

负责从飞书多维表格读取选手提交信息、回写评测分数与状态。
使用飞书开放平台 Lark Open API，认证方式为 Tenant Access Token。

飞书表格列结构（实际约定）:
    记录ID | 队伍编号 | 队伍名称 | 队长邮箱 | 镜像名称 | 评审状态 | 得分

状态值（中文）:
    评测中 / 评测完成 / 评测失败 / 镜像异常

使用示例:

```python
from lerobot.sim_eval.spreadsheet import (
    CompetitorInfo,
    EvalScore,
    FeishuBitableClient,
)

client = FeishuBitableClient(
    app_id="cli_xxx",
    app_secret="xxx",
    bitable_app_token="xxx",
    table_id="tblxxx",
)
competitors = client.fetch_competitors()
for c in competitors:
    client.update_score(c, EvalScore(score=85.5, status="评测完成"))
```
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from .registry import validate_image_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 飞书 Open API 端点
# ---------------------------------------------------------------------------
_FEISHU_AUTH_URL = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
_FEISHU_BITABLE_RECORDS_URL = (
    "https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records"
)
_FEISHU_BITABLE_RECORD_URL = (
    "https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/{record_id}"
)

# ---------------------------------------------------------------------------
# 飞书多维表格列名（与表格结构严格一致）
# ---------------------------------------------------------------------------
COL_TEAM_ID = "队伍编号"
COL_TEAM_NAME = "队伍名称"
COL_EMAIL = "队长邮箱"
COL_IMAGE = "镜像名称"
COL_STATUS = "评审状态"
COL_SCORE = "得分"

# 状态值（中文，与飞书表格"评审状态"列的选项一致）
STATUS_EVALUATING = "评测中"
STATUS_DONE = "评测完成"
STATUS_FAILED = "评测失败"
STATUS_IMAGE_ERROR = "镜像异常"


@dataclass
class CompetitorInfo:
    """选手提交信息，对应飞书表格中的一条记录。

    Attributes:
        record_id: 飞书表格记录 ID（用于回写更新）。
        team_id: 队伍编号。
        name: 队伍名称。
        email: 队长邮箱。
        image_name: Docker 镜像地址。
        status: 当前评审状态。
    """

    record_id: str
    team_id: str = ""
    name: str = ""
    email: str = ""
    image_name: str = ""
    status: str = STATUS_EVALUATING


@dataclass
class EvalScore:
    """单次评测的分数与状态。

    Attributes:
        score: 得分（0-100）。
        status: 评测结果状态。
        detail: 详细描述（失败原因、各任务细项等，写入飞书备注或单独列）。
    """

    score: float = 0.0
    status: str = STATUS_DONE
    detail: str = ""


@dataclass
class _TokenCache:
    """租户访问令牌缓存。"""

    token: str = ""
    expires_at: float = 0.0


class FeishuBitableClient:
    """飞书多维表格客户端。

    通过 Lark Open API 读取/更新多维表格中的选手记录。
    内部自动管理 tenant_access_token 的获取与刷新。
    """

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        bitable_app_token: str,
        table_id: str,
        page_size: int = 100,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._bitable_app_token = bitable_app_token
        self._table_id = table_id
        self._page_size = page_size
        self._token_cache = _TokenCache()

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def fetch_competitors(
        self,
        status_filter: list[str] | None = None,
    ) -> list[CompetitorInfo]:
        """从多维表格获取待评测选手列表。

        Args:
            status_filter: 要筛选的状态列表，默认取"评测中"的记录。

        Returns:
            CompetitorInfo 列表。
        """
        if status_filter is None:
            status_filter = [STATUS_EVALUATING]

        records = self._fetch_all_records()
        competitors: list[CompetitorInfo] = []

        for record in records:
            fields = record.get("fields", {})
            status = self._get_text_field(fields, COL_STATUS)

            # 如果状态列为空视为待评测（首次提交）
            effective_status = status or STATUS_EVALUATING
            if effective_status not in status_filter:
                continue

            image_name = self._get_text_field(fields, COL_IMAGE)
            if not image_name:
                # 空行或模板行，跳过（不打印 warning 避免日志噪音）
                logger.debug("跳过记录 %s：镜像名称为空", record.get("record_id"))
                continue

            competitors.append(
                CompetitorInfo(
                    record_id=record["record_id"],
                    team_id=self._get_text_field(fields, COL_TEAM_ID),
                    name=self._get_text_field(fields, COL_TEAM_NAME),
                    email=self._get_text_field(fields, COL_EMAIL),
                    image_name=image_name,
                    status=effective_status,
                )
            )

        logger.info("从飞书表格获取到 %s 条待评测记录", len(competitors))
        return competitors

    def update_score(self, competitor: CompetitorInfo, score: EvalScore) -> None:
        """将评测结果写回多维表格。

        更新"评审状态"和"得分"两列。

        Args:
            competitor: 选手信息。
            score: 评测分数与状态。
        """
        fields: dict[str, Any] = {
            COL_STATUS: score.status,
            COL_SCORE: str(score.score),  # 飞书字段是文本类型，必须传字符串
        }
        self._update_record(competitor.record_id, fields)
        logger.info(
            "回写 %s: score=%.1f status=%s",
            competitor.name or competitor.team_id, score.score, score.status,
        )

    @staticmethod
    def validate_image_name(image_name: str) -> bool:
        """校验镜像名，委托给 registry.validate_image_name。"""
        return validate_image_name(image_name)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        """获取有效的 tenant_access_token，过期自动刷新。"""
        if self._token_cache.token and time.time() < self._token_cache.expires_at - 60:
            return self._token_cache.token

        resp = requests.post(
            _FEISHU_AUTH_URL,
            json={"app_id": self._app_id, "app_secret": self._app_secret},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"飞书鉴权失败：code={data.get('code')} msg={data.get('msg')}")

        self._token_cache.token = data["tenant_access_token"]
        self._token_cache.expires_at = time.time() + data.get("expire", 7200)
        return self._token_cache.token

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json; charset=utf-8",
        }

    def _fetch_all_records(self) -> list[dict[str, Any]]:
        """分页获取多维表格全部记录。"""
        all_records: list[dict[str, Any]] = []
        page_token: str | None = None

        while True:
            params: dict[str, Any] = {"page_size": self._page_size}
            if page_token:
                params["page_token"] = page_token

            resp = requests.get(
                _FEISHU_BITABLE_RECORDS_URL.format(
                    app_token=self._bitable_app_token,
                    table_id=self._table_id,
                ),
                headers=self._headers(),
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                raise RuntimeError(
                    f"飞书读取表格失败：code={data.get('code')} msg={data.get('msg')}"
                )

            items = data.get("data", {}).get("items", [])
            all_records.extend(items)

            if not data.get("data", {}).get("has_more"):
                break
            page_token = data["data"].get("page_token")

        return all_records

    def _update_record(self, record_id: str, fields: dict[str, Any]) -> None:
        """更新单条记录。"""
        url = _FEISHU_BITABLE_RECORD_URL.format(
            app_token=self._bitable_app_token,
            table_id=self._table_id,
            record_id=record_id,
        )
        resp = requests.put(
            url,
            headers=self._headers(),
            json={"fields": fields},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(
                f"飞书更新记录失败：code={data.get('code')} msg={data.get('msg')} "
                f"url={url} body={json.dumps({'fields': fields})}"
            )

    @staticmethod
    def _get_text_field(fields: dict[str, Any], key: str) -> str:
        """从飞书字段值中提取纯文本。

        飞书文本字段可能是:
        - 纯字符串: "hello"
        - 富文本数组: [{"type": "text", "text": "hello"}]
        - 链接对象: {"link": "http://...", "text": "display text"}
        - 数字: 42
        """
        value = fields.get(key, "")
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            # 飞书链接/附件字段: {"link": "...", "text": "..."}
            return value.get("text", "") or value.get("link", "") or ""
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        if isinstance(value, (int, float)):
            return str(value)
        return ""
