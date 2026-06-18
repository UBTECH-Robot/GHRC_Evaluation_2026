"""Docker 镜像仓库客户端。

负责 Docker 登录、拉取选手镜像、校验与清理。
支持 Docker Hub、ACR、Harbor 等兼容 OCI 的镜像仓库。
所有操作通过 subprocess 调用 docker CLI 完成。

使用示例:

```python
from lerobot.sim_eval.registry import DockerRegistryClient

client = DockerRegistryClient(
    registry="docker.io",
    username="your-username",
    password="your-password",
)
client.login()
client.pull_image("docker.io/team/repo:v1.0")
```

安全注意事项:
    镜像名会经过正则白名单校验，防止命令注入。
    密码通过 stdin 传入，不会打印到日志或控制台。
"""

from __future__ import annotations

import functools
import logging
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 镜像名校验 — 支持 Docker Hub 及任意 OCI registry
# ---------------------------------------------------------------------------
_IMAGE_NAME_RE = re.compile(
    r"^"
    r"[a-zA-Z0-9][a-zA-Z0-9_.\-]*"     # registry 域名部分
    r"(:[0-9]+)?"                       # 可选端口
    r"(/[a-zA-Z0-9_.\-]+)+"             # /namespace/repo 路径（至少一段）
    r"(:[a-zA-Z0-9_.\-]+)?"             # 可选 :tag
    r"(@sha256:[a-f0-9]{64})?"          # 可选 @digest
    r"$"
)

_DOCKER_TIMEOUT = 600       # pull 超时（大镜像需要较长时间）
_DOCKER_CMD_TIMEOUT = 30    # 普通命令超时
_MIN_FREE_DISK_GB = 10      # 最小可用磁盘空间（GB）

F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    base_delay: float = 10.0,
    backoff: float = 2.0,
    max_delay: float = 120.0,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[F], F]:
    """通用重试装饰器，指数退避。

    Args:
        max_attempts: 最大尝试次数（含首次）。
        base_delay: 首次重试等待秒数。
        backoff: 退避倍数。
        max_delay: 最大等待秒数。
        on_retry: 每次重试前的回调，接收 (exception, attempt_number)。
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt >= max_attempts:
                        raise
                    delay = min(base_delay * (backoff ** (attempt - 1)), max_delay)
                    logger.warning(
                        "%s 失败 (attempt %s/%s)，%ss 后重试: %s",
                        func.__name__, attempt, max_attempts, delay, exc,
                    )
                    if on_retry:
                        on_retry(exc, attempt)
                    time.sleep(delay)

            raise RuntimeError("unreachable") from last_exc

        return wrapper  # type: ignore[return-value]

    return decorator


def check_disk_space(min_free_gb: int = _MIN_FREE_DISK_GB) -> bool:
    """检查磁盘可用空间是否充足。

    Returns:
        True 表示空间充足。
    """
    usage = shutil.disk_usage("/var/lib/docker")
    free_gb = usage.free / (1024**3)
    ok = free_gb >= min_free_gb
    if not ok:
        logger.error("磁盘空间不足：可用 %.1fGB，需要 >= %dGB", free_gb, min_free_gb)
    else:
        logger.info("磁盘空间: %.1fGB 可用", free_gb)
    return ok


def validate_image_name(image_name: str) -> bool:
    """校验镜像名是否为合法 Docker 镜像地址。

    支持的格式:
        docker.io/namespace/repo:tag
        registry.example.com:5000/ns/repo:tag
        namespace/repo:tag          (隐含 docker.io)
        repo:tag                    (隐含 docker.io/library)

    Args:
        image_name: 待校验的镜像名。

    Returns:
        True 表示格式合法。
    """
    return bool(_IMAGE_NAME_RE.match(image_name))


class DockerRegistryClient:
    """Docker 镜像仓库客户端。

    封装 docker login / pull / rmi 操作，包含镜像名校验与重试机制。
    支持 Docker Hub 及任意 OCI 兼容 registry。
    """

    def __init__(self, registry: str, username: str, password: str) -> None:
        """初始化 Docker registry 客户端。

        Args:
            registry: Registry 地址。
                Docker Hub 使用 "docker.io"（或不传，默认）。
                ACR 使用 "registry.cn-hangzhou.aliyuncs.com"。
            username: 登录用户名。
            password: 登录密码（不会记录到日志）。
        """
        self._registry = registry
        self._username = username
        self._password = password
        self._logged_in = False

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def login(self) -> None:
        """登录镜像仓库。

        命令: docker login --username <user> --password-stdin [registry]

        Raises:
            RuntimeError: 登录失败时抛出。
        """
        target = self._registry if self._registry and self._registry != "docker.io" else ""
        cmd = ["docker", "login", "--username", self._username, "--password-stdin"]
        if target:
            cmd.append(target)

        logger.info("登录 registry：%s", target or "docker.io (默认)")
        try:
            proc = subprocess.run(
                cmd,
                input=self._password,
                capture_output=True,
                text=True,
                timeout=_DOCKER_CMD_TIMEOUT,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Docker 登录失败：{proc.stderr.strip()}")
            self._logged_in = True
            logger.info("登录成功")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Docker 登录超时：{target}")

    def pull_image(self, image_name: str, retries: int = 2) -> None:
        """拉取 Docker 镜像。

        Args:
            image_name: 完整的镜像地址（含 tag 或 digest）。
            retries: 失败重试次数。

        Raises:
            ValueError: 镜像名格式不合法。
            RuntimeError: 拉取最终失败。
        """
        if not validate_image_name(image_name):
            raise ValueError(f"镜像名格式不合法：{image_name}")

        if not self._logged_in:
            self.login()

        logger.info("拉取镜像：%s", image_name)
        last_error: Exception | None = None

        for attempt in range(1, retries + 2):
            try:
                _run_docker(["pull", image_name], timeout=_DOCKER_TIMEOUT)
                logger.info("镜像拉取完成：%s", image_name)
                return
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                last_error = exc
                logger.warning(
                    "镜像拉取失败 (attempt %s/%s)：%s", attempt, retries + 1, exc
                )
                if attempt <= retries:
                    wait = min(30 * attempt, 120)
                    logger.info("等待 %s 秒后重试...", wait)
                    time.sleep(wait)

        raise RuntimeError(f"镜像拉取最终失败：{image_name}") from last_error

    def image_exists(self, image_name: str) -> bool:
        """检查本地是否已有指定镜像。"""
        try:
            _run_docker(["image", "inspect", image_name], timeout=_DOCKER_CMD_TIMEOUT)
            return True
        except subprocess.CalledProcessError:
            return False

    def remove_image(self, image_name: str) -> bool:
        """删除本地镜像（节省磁盘空间）。

        Returns:
            True 表示删除成功。
        """
        if not validate_image_name(image_name):
            logger.warning("跳过删除，镜像名格式不合法：%s", image_name)
            return False

        try:
            _run_docker(["rmi", image_name], timeout=_DOCKER_CMD_TIMEOUT)
            logger.info("已删除本地镜像：%s", image_name)
            return True
        except subprocess.CalledProcessError as exc:
            logger.warning("删除镜像失败（可能被其他容器引用）：%s", exc)
            return False

    def get_image_digest(self, image_name: str) -> str:
        """获取镜像的仓库摘要（用于记录/校验）。

        Returns:
            镜像摘要字符串，获取失败时返回空字符串。
        """
        try:
            result = _run_docker(
                ["image", "inspect", image_name, "--format", "{{index .RepoDigests 0}}"],
                timeout=_DOCKER_CMD_TIMEOUT,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return ""


def _run_docker(
    args: list[str], timeout: int = _DOCKER_CMD_TIMEOUT
) -> subprocess.CompletedProcess[str]:
    """封装 subprocess.run 调用 docker CLI。"""
    cmd = ["docker"] + args
    logger.debug("执行 docker 命令：%s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
