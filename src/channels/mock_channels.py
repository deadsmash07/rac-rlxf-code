"""Mock channels for unit / integration testing without verl or judge APIs.

Registered under "mock_sync" and "mock_async" so YAML configs can reference
them without requiring the real code_exec or livecodebench_judge runtimes.
"""
from __future__ import annotations

import torch

from .registry import register_channel


@register_channel("mock_sync")
class MockSyncChannel:
    """Returns a deterministic scalar reward from meta_info."""

    is_async = False

    def __init__(self, lambda_weight: float = 1.0, default_reward: float = 0.5):
        self.lambda_weight = lambda_weight
        self.default_reward = default_reward

    def __call__(self, data):
        r = self.default_reward
        if hasattr(data, "meta_info") and "reward_fast" in data.meta_info:
            r = float(data.meta_info["reward_fast"])
        return torch.tensor(r).unsqueeze(0)


@register_channel("mock_async")
class MockAsyncChannel:
    """Always-ready async channel returning a constant reward."""

    is_async = True

    def __init__(self, lambda_weight: float = 1.0, expected_delay_steps: int = 10,
                 default_reward: float = 0.8):
        self.lambda_weight = lambda_weight
        self.expected_delay_steps = expected_delay_steps
        self.default_reward = default_reward
        self._tasks: dict[str, int] = {}

    def submit(self, data):
        tid = f"mock-{len(self._tasks)}"
        step = 0
        if hasattr(data, "meta_info") and "global_step" in data.meta_info:
            step = int(data.meta_info["global_step"])
        self._tasks[tid] = step
        return tid

    def try_fetch(self, tid):
        return torch.tensor(self.default_reward).unsqueeze(0)
