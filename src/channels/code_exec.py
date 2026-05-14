"""Fast-channel reward: code execution against unit tests.

Sync channel (~1-5s latency). Reward = fraction of test cases passing on the
generated code. For LiveCodeBench integration, each problem has a set of
stdin/stdout test pairs — we run `python3 -c <generated>` with each stdin
and compare stdout.

In production, verl calls this each training step with batched rollouts.
"""
from __future__ import annotations

import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch

from .registry import register_channel


@register_channel("code_exec")
class CodeExecChannel:
    """Fast sync reward from running unit tests on generated code.

    Config (from YAML):
        max_latency_s: int    # per-test timeout (default 5)
        lambda_weight: float  # Λ[k, Δ=0] weight (consumed by reward manager)
    """

    is_async = False

    def __init__(
        self,
        max_latency_s: int = 5,
        lambda_weight: float = 1.0,
        python_exe: str = "python3",
    ):
        self.max_latency_s = max_latency_s
        self.lambda_weight = lambda_weight
        self.python_exe = python_exe

    def __call__(self, data: Any) -> torch.Tensor:
        """Compute per-rollout fast reward.

        `data` is verl's DataProto with expected keys:
            - `completion_texts`: list[str] generated code per rollout
            - `test_cases`: list[dict{stdin, stdout}] per prompt
        Returns (batch,) reward tensor.
        """
        # Extract completion texts (verl-compatible access pattern)
        completions = self._extract_completions(data)
        test_cases = self._extract_test_cases(data)

        rewards = torch.zeros(len(completions))
        for i, (code, cases) in enumerate(zip(completions, test_cases)):
            if not cases:
                rewards[i] = 0.0
                continue
            passed = sum(1 for case in cases if self._run_one(code, case))
            rewards[i] = passed / len(cases)
        return rewards

    def _extract_completions(self, data) -> list[str]:
        if hasattr(data, "non_tensor_batch") and "completion" in data.non_tensor_batch:
            return list(data.non_tensor_batch["completion"])
        if hasattr(data, "completion_texts"):
            return list(data.completion_texts)
        if hasattr(data, "meta_info") and "completion_text" in data.meta_info:
            return [data.meta_info["completion_text"]]
        return []

    def _extract_test_cases(self, data) -> list[list[dict]]:
        if hasattr(data, "non_tensor_batch") and "test_cases" in data.non_tensor_batch:
            return list(data.non_tensor_batch["test_cases"])
        if hasattr(data, "test_cases"):
            return data.test_cases
        if hasattr(data, "meta_info") and "test_cases" in data.meta_info:
            return [data.meta_info["test_cases"]]
        return [[] for _ in self._extract_completions(data)]

    def _run_one(self, code: str, case: dict) -> bool:
        """Execute `code` with `case["stdin"]`; compare to `case["stdout"]`."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            script_path = f.name
        try:
            result = subprocess.run(
                [self.python_exe, script_path],
                input=case.get("stdin", ""),
                capture_output=True,
                text=True,
                timeout=self.max_latency_s,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        finally:
            Path(script_path).unlink(missing_ok=True)

        if result.returncode != 0:
            return False
        expected = case.get("stdout", "").strip()
        got = result.stdout.strip()
        return expected == got
