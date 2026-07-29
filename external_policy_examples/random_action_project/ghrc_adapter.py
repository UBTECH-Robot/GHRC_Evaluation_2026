"""GHRC adapter for a migrated external random-action project."""

from __future__ import annotations

from typing import Any

from src.lerobot.sim_eval.policy_adapter import InferenceContext, PolicyAdapter, ResetContext

from .external_algo.random_network import RandomActionNetwork


class ExternalRandomPolicyAdapter(PolicyAdapter):
    """Wrap an external network object with the GHRC `PolicyAdapter` API."""

    def __init__(self) -> None:
        self.network: RandomActionNetwork | None = None

    def load(self, model_path: str, device: str, config: dict[str, Any]) -> None:
        """Initialize the external algorithm.

        Args:
            model_path: Optional checkpoint path. This demo does not use it.
            device: Runtime device requested by YAML. This demo is CPU-only.
            config: `adapter_config` from YAML.
        """

        del model_path, device
        self.network = RandomActionNetwork(
            action_dim=int(config.get("action_dim", 20)),
            action_low=float(config.get("action_low", -1.0)),
            action_high=float(config.get("action_high", 1.0)),
            seed=config.get("seed"),
        )

    def predict(
        self,
        observation: dict[str, Any],
        context: InferenceContext,
    ) -> list[float]:
        """Run one step of external inference and return a 1D action."""

        del context
        if self.network is None:
            raise RuntimeError("ExternalRandomPolicyAdapter has not been loaded")
        return self.network.forward(observation)

    def reset(self, reset_context: ResetContext | None = None) -> None:
        """Forward episode reset to the external algorithm."""

        del reset_context
        if self.network is not None:
            self.network.reset()

    def close(self) -> None:
        """Release external algorithm resources."""

        self.network = None
