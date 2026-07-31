"""A stand-in network from an external project.

In a real migration, this file would be replaced by the participant's own
model code. The GHRC-specific wrapper should stay in `ghrc_adapter.py`.
"""

from __future__ import annotations

import random
from typing import Any


class RandomActionNetwork:
    """Minimal network-like object that returns random continuous actions."""

    def __init__(
        self,
        action_dim: int = 20,
        action_low: float = -1.0,
        action_high: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if action_high < action_low:
            raise ValueError(
                f"action_high must be greater than or equal to action_low, got {action_high} < {action_low}"
            )

        self.action_dim = int(action_dim)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self._rng = random.Random(seed)

    def forward(self, observation: dict[str, Any]) -> list[float]:
        """Return a random action vector.

        Args:
            observation: LeRobot-format observation. This demo ignores it.
        """

        del observation
        return [
            self._rng.uniform(self.action_low, self.action_high)
            for _ in range(self.action_dim)
        ]

    def reset(self) -> None:
        """Reset network state.

        This random demo has no episode state. Real algorithms can clear RNN
        hidden states, action queues, history buffers, or planner state here.
        """
