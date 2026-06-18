"""Zero-action custom policy adapter example.

This module is a minimal reference for participants who need to implement a
custom `PolicyAdapter`. It ignores observations and always returns an all-zero
action vector.
"""

from __future__ import annotations

from typing import Any

from src.lerobot.sim_eval.policy_adapter import InferenceContext, PolicyAdapter, ResetContext


class ZeroActionPolicyAdapter(PolicyAdapter):
    """A minimal custom adapter that always outputs zero actions."""

    def __init__(self) -> None:
        self.action_dim = 20
        self._action = [0.0] * self.action_dim

    def load(self, model_path: str, device: str, config: dict[str, Any]) -> None:
        """Load runtime config for the zero-action policy.

        Args:
            model_path: Unused. Kept for compatibility with `PolicyAdapter`.
            device: Unused. Kept for compatibility with `PolicyAdapter`.
            config: Optional config. Supports `action_dim`, defaulting to 20.
        """

        del model_path, device
        action_dim = int(config.get("action_dim", 20))
        if action_dim <= 0:
            raise ValueError(f"action_dim 必须为正整数，收到：{action_dim}")
        self.action_dim = action_dim
        self._action = [0.0] * self.action_dim

    def predict(
        self,
        observation: dict[str, Any],
        context: InferenceContext,
    ) -> list[float]:
        """Return an all-zero action vector for every observation."""

        del observation, context
        return list(self._action)

    def reset(self, reset_context: ResetContext | None = None) -> None:
        """No internal state is kept, so reset is a no-op."""

        del reset_context

    def close(self) -> None:
        """No resources are held, so close is a no-op."""
