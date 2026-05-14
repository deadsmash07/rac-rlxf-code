"""GSM8K reward channels for the full-scale real-LLM RAC vs vanilla experiment.

Two channel classes:

    GSM8KFastChannel       — sync, returns GSM8K exact-match reward immediately.
    GSM8KDelayedChannel    — async, enforces a configurable Δ-step delay so the
                              RAC forward-injection path can exercise real
                              dynamics at Δ>=1.

Both consume verl's canonical GSM8K scorer:
    from verl.utils.reward_score.gsm8k import compute_score

In verl's async v1 reward path, the manager calls `run_single(data)` once per
sample. That path uses the FAST channel exclusively. For the delayed arm we
DO NOT use the slow channel for the per-sample score (verl would block on
it); instead the "delay" is modelled by the reward manager's `__call__`
path, which stores ROLLOUT records and delivers the same reward at step
t+Δ via RAC forward-injection. For the purposes of the real-LLM A/B, we
therefore run the DELAYED arm by:

  1. Making the FAST channel return a CONSTANT baseline (e.g. 0) so the
     per-sample verl reward path has zero immediate signal.
  2. Making the SLOW channel carry the REAL GSM8K pass/fail, returnable
     only after Δ optimizer steps via `try_fetch`.
  3. The RAC correction δ = α · (r_slow − r_fast_baseline) then lands on
     the NEXT step's advantage via _flush_forward_injection, exactly the
     semantics the paper claims.

The VANILLA arm runs with GSM8KFastChannel only — reward delivered
immediately with no delay and no RAC correction.

Why this is the right experiment for §4.2:
  * Both arms see the SAME aggregate reward distribution.
  * Only the delivery-delay structure differs.
  * Any difference in learning trajectory is attributable to RAC.
  * Matches the Δ=5 setting the paper's MDP proof operates under.
"""
from __future__ import annotations

from typing import Any
import os

import torch

from .registry import register_channel


def _import_gsm8k_scorer():
    """Import verl's canonical GSM8K compute_score, with a fallback if verl
    is absent (for unit-test environments)."""
    try:
        from verl.utils.reward_score.gsm8k import compute_score as _cs
        return _cs
    except Exception:
        # Minimal fallback: exact-match on `#### <number>` pattern.
        import re

        def _cs(solution_str: str, ground_truth: str,
               method: str = "strict", format_score: float = 0.0,
               score: float = 1.0) -> float:
            m = re.findall(r"#### (-?[0-9.,]+)", solution_str or "")
            if not m:
                return 0.0
            pred = m[-1].replace(",", "").replace("$", "")
            return float(score) if pred == str(ground_truth).strip() else float(format_score)

        return _cs


_compute_score = _import_gsm8k_scorer()


def _extract_completion_and_gt(data: Any) -> list[tuple[str, str]]:
    """Extract (response_text, ground_truth) per row from a verl DataProto-ish.

    Robust to several verl shapes we have seen on the smoke path:
      - data.non_tensor_batch["completion"] + ["reward_model"]["ground_truth"]
      - per-row ntb dict with ["extra_info"]["answer"]
      - shim with ["reward_model"] (list of dicts)
    """
    ntb = getattr(data, "non_tensor_batch", None) or {}
    completions = ntb.get("completion") or ntb.get("response") or []
    rms = ntb.get("reward_model") or [{}] * len(completions)
    out: list[tuple[str, str]] = []
    for i, comp in enumerate(completions):
        rm = rms[i] if i < len(rms) else {}
        gt = ""
        if isinstance(rm, dict):
            gt = rm.get("ground_truth", "") or ""
        out.append((comp or "", str(gt)))
    return out


@register_channel("gsm8k_fast")
class GSM8KFastChannel:
    """Synchronous GSM8K exact-match reward.

    Config:
        lambda_weight : float — Λ[k=0, Δ=0] weight (consumed downstream)
        method        : "strict" or "flexible"  — passed to verl's compute_score
        format_score  : float — reward if model produced a number but wrong
        score         : float — reward if answer matches ground truth
        constant_override : float | None — if set, returns this constant
            value for every sample (used to build the "zero fast, real slow"
            DELAYED arm where the real signal travels via the slow channel).
    """

    is_async = False

    def __init__(
        self,
        lambda_weight: float = 1.0,
        method: str = "strict",
        format_score: float = 0.0,
        score: float = 1.0,
        constant_override: float | None = None,
    ):
        self.lambda_weight = float(lambda_weight)
        self.method = method
        self.format_score = float(format_score)
        self.score = float(score)
        self.constant_override = (
            None if constant_override is None else float(constant_override)
        )

    def __call__(self, data: Any) -> torch.Tensor:
        pairs = _extract_completion_and_gt(data)
        if self.constant_override is not None:
            return torch.full((max(len(pairs), 1),), self.constant_override)
        rewards = []
        for comp, gt in pairs:
            if not gt:
                rewards.append(0.0)
                continue
            r = _compute_score(
                solution_str=comp,
                ground_truth=gt,
                method=self.method,
                format_score=self.format_score,
                score=self.score,
            )
            rewards.append(float(r))
        if not rewards:
            rewards = [0.0]
        return torch.tensor(rewards, dtype=torch.float32)

    # run_single support — the reward manager also routes here per-sample
    # through its fallback inside run_single.
    def score_single(self, response_str: str, ground_truth: str) -> float:
        if self.constant_override is not None:
            return float(self.constant_override)
        if not ground_truth:
            return 0.0
        return float(_compute_score(
            solution_str=response_str,
            ground_truth=ground_truth,
            method=self.method,
            format_score=self.format_score,
            score=self.score,
        ))


@register_channel("gsm8k_delayed")
class GSM8KDelayedChannel:
    """Async GSM8K exact-match reward with configurable Δ-step delay.

    The channel immediately computes the reward on `submit()`, stashes it
    keyed by task-id alongside the submitter's global_step, and refuses to
    deliver it via `try_fetch` until `delay_steps` optimizer steps have
    elapsed (tracked by `self.current_step` which the reward manager
    updates each __call__).

    Config:
        delay_steps   : int    — Δ (default 5)
        lambda_weight : float
        method, format_score, score — same as GSM8KFastChannel
    """

    is_async = True

    def __init__(
        self,
        delay_steps: int = 5,
        lambda_weight: float = 1.0,
        method: str = "strict",
        format_score: float = 0.0,
        score: float = 1.0,
        expected_delay_steps: int | None = None,  # accepted for YAML compat
        max_latency_s: int | None = None,         # accepted for YAML compat
    ):
        self.delay_steps = int(delay_steps if expected_delay_steps is None
                               else expected_delay_steps)
        self.lambda_weight = float(lambda_weight)
        self.method = method
        self.format_score = float(format_score)
        self.score = float(score)
        self.max_latency_s = max_latency_s
        # task_id -> (submit_step, reward_tensor)
        self._pending: dict[str, tuple[int, torch.Tensor]] = {}
        self.current_step: int = 0

    def tick(self, step: int) -> None:
        """Called by the reward manager each __call__ to advance 'time'."""
        self.current_step = int(step)

    def submit(self, data: Any) -> str:
        pairs = _extract_completion_and_gt(data)
        rewards = []
        for comp, gt in pairs:
            if not gt:
                rewards.append(0.0)
                continue
            r = _compute_score(
                solution_str=comp,
                ground_truth=gt,
                method=self.method,
                format_score=self.format_score,
                score=self.score,
            )
            rewards.append(float(r))
        if not rewards:
            rewards = [0.0]
        tid = f"gsm8k-delayed-{len(self._pending)}-{self.current_step}"
        self._pending[tid] = (
            self.current_step,
            torch.tensor(rewards, dtype=torch.float32),
        )
        return tid

    def try_fetch(self, tid: str):
        if tid not in self._pending:
            return None
        submit_step, r = self._pending[tid]
        if self.current_step - submit_step < self.delay_steps:
            return None  # still "in flight"
        # Deliver and clear
        del self._pending[tid]
        return r
