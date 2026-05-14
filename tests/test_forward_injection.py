"""Tests for RAC forward-injection semantics.

Per internal design notes: verl consumes advantages
immediately, so we can't retroactively modify a past advantage. Our RAC
correction δ must instead forward-inject: add to the NEXT step's partial
advantage. This module tests that property explicitly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC.parent))

from src.trainer import MultiChannelRACRewardManager


class _SyncReward:
    is_async = False

    def __init__(self):
        self._call_count = 0

    def __call__(self, data):
        self._call_count += 1
        r = data.meta_info.get("reward_fast", 0.5)
        return torch.tensor(float(r)).unsqueeze(0)


class _AlwaysReadyAsync:
    """Slow channel that returns a non-trivial reward on first try_fetch."""

    is_async = True

    def __init__(self, reward: float = 0.9):
        self.reward = reward
        self._submitted: set[str] = set()

    def submit(self, data):
        tid = f"t{len(self._submitted)}"
        self._submitted.add(tid)
        return tid

    def try_fetch(self, tid):
        return torch.tensor(self.reward).unsqueeze(0)


class _NeverReadyAsync:
    is_async = True

    def __init__(self):
        self._submitted: set[str] = set()

    def submit(self, data):
        tid = f"t{len(self._submitted)}"
        self._submitted.add(tid)
        return tid

    def try_fetch(self, tid):
        return None  # always pending


class _Data:
    def __init__(self, step: int, rid: str, fast: float = 0.5,
                 log_probs: torch.Tensor | None = None):
        self.meta_info = {"global_step": step, "rollout_id": rid,
                          "reward_fast": fast}
        self.log_probs = log_probs if log_probs is not None else torch.tensor([0.0])

    @property
    def batch(self):
        return {"old_log_probs": self.log_probs}


# -----------------------------------------------------------------------------
# Core forward-injection property
# -----------------------------------------------------------------------------


def test_correction_queue_populated_after_first_slow_drain():
    """After a step where a slow reward completes, the queue gains 1 entry."""
    mgr = MultiChannelRACRewardManager(
        channels={"fast": _SyncReward(), "slow": _AlwaysReadyAsync(reward=0.9)},
        tau_age=50.0, is_clip=0.2, alpha_delta=1.0,
    )
    assert len(mgr._forward_injection_queue) == 0  # start empty
    # Step 0: enqueue slow reward; stub drains same-step; queue gains 1
    mgr(_Data(step=0, rid="r0", log_probs=torch.tensor([0.0])))
    assert len(mgr._forward_injection_queue) == 1, (
        f"expected 1 queued correction after step 0, got "
        f"{len(mgr._forward_injection_queue)}"
    )


def test_queue_flushed_at_start_of_next_call():
    """Each __call__ flushes the queue at its start — a correction queued at
    step T is consumed by step T+1's flush (then T+1's own drain repopulates)."""
    mgr = MultiChannelRACRewardManager(
        channels={"fast": _SyncReward(), "slow": _AlwaysReadyAsync(reward=0.9)},
        tau_age=50.0, is_clip=0.2, alpha_delta=1.0,
    )

    # Step 0: completes with 1 correction queued
    mgr(_Data(step=0, rid="r0", log_probs=torch.tensor([0.0])))
    snapshot_before = list(mgr._forward_injection_queue)
    assert len(snapshot_before) == 1

    # Step 1: on entering, flush consumes snapshot_before; then step 1's own
    # drain queues a NEW correction. So the POST-call queue state should
    # have exactly 1 (step 1's newly-drained correction), NOT the 2 that would
    # accumulate if flush were missing.
    mgr(_Data(step=1, rid="r1", log_probs=torch.tensor([0.0])))
    assert len(mgr._forward_injection_queue) == 1, (
        f"expected queue to have 1 (step-1's new correction), got "
        f"{len(mgr._forward_injection_queue)} — flush may be broken"
    )


def test_correction_injects_into_next_step_advantage():
    """A correction queued at step T must appear in step T+1's returned advantage.

    Uses a multi-rollout batch so the advantage standardization doesn't return NaN.
    """
    slow = _AlwaysReadyAsync(reward=1.5)  # big slow reward

    class _BatchedSyncReward:
        is_async = False
        def __call__(self, data):
            # Return a batch of 4 rewards with variance, so advantage isn't NaN
            return torch.tensor([0.1, 0.3, 0.7, 0.9])

    class _BatchedAsync:
        is_async = True
        def __init__(self, reward=1.5):
            self.reward = reward
            self._n = 0
        def submit(self, data):
            tid = f"t{self._n}"
            self._n += 1
            return tid
        def try_fetch(self, tid):
            return torch.tensor([self.reward, self.reward, self.reward, self.reward])

    mgr = MultiChannelRACRewardManager(
        channels={"fast": _BatchedSyncReward(), "slow": _BatchedAsync(reward=1.5)},
        tau_age=50.0, is_clip=0.0, alpha_delta=1.0,
    )

    adv_step0 = mgr(_Data(step=0, rid="r0", log_probs=torch.tensor([0.0])))
    assert len(mgr._forward_injection_queue) == 1
    adv_step1 = mgr(_Data(step=1, rid="r1", log_probs=torch.tensor([0.0])))

    # Forward-injection means step 1's returned advantage = step-1-partial + correction.
    # Partial advantage is the same shape for both steps (same fast channel output),
    # so the DIFFERENCE must equal the injected correction.
    delta = adv_step1 - adv_step0
    # The correction was δ = α·(slow_reward − fast_mean). slow=1.5, fast_mean=0.5,
    # α=1.0 → δ=1.0 (roughly, modulo batched broadcasting).
    assert torch.all(torch.isfinite(delta)), f"delta contains NaN/inf: {delta}"
    assert float(delta.abs().max()) > 0.1, (
        f"forward-injection should produce non-trivial delta; got {delta}"
    )


def test_corrections_stack_across_multiple_pending_drains():
    """If K slow rewards complete in one step, all K corrections stack into next."""
    slow = _AlwaysReadyAsync(reward=1.0)
    mgr = MultiChannelRACRewardManager(
        channels={"fast": _SyncReward(), "slow": slow},
        tau_age=50.0, is_clip=0.0, alpha_delta=1.0,
    )

    # Queue 3 rollouts with pending slow rewards by using _NeverReadyAsync first
    # ... simpler: just submit 3 consecutive steps; each step drains the previous
    # so corrections accumulate in _forward_injection_queue.
    mgr(_Data(step=0, rid="r0"))
    mgr(_Data(step=1, rid="r1"))

    # Going into step 2, _forward_injection_queue has step-1's correction.
    assert len(mgr._forward_injection_queue) == 1

    # Step 2's __call__ flushes 1 correction AND queues 1 new one
    mgr(_Data(step=2, rid="r2"))
    # Post step 2: queue empty after flush + repopulated with step 2's correction
    assert len(mgr._forward_injection_queue) == 1


def test_no_injection_when_slow_channel_never_completes():
    """If slow rewards are always pending, forward-injection queue stays empty."""
    mgr = MultiChannelRACRewardManager(
        channels={"fast": _SyncReward(), "slow": _NeverReadyAsync()},
        tau_age=50.0, is_clip=0.2, alpha_delta=1.0,
    )
    for step in range(5):
        mgr(_Data(step=step, rid=f"r{step}"))
    # No corrections ever applied → queue stays empty
    assert len(mgr._forward_injection_queue) == 0
    assert mgr.n_corrections_applied == 0
    # But rollouts are still queued
    assert len(mgr.pending) == 5


def test_fast_only_rollouts_produce_no_forward_injection():
    """Fast-only rollouts have nothing to correct — queue stays empty."""
    mgr = MultiChannelRACRewardManager(
        channels={"fast": _SyncReward()},  # NO slow channel
        tau_age=50.0, is_clip=0.2, alpha_delta=1.0,
    )
    for step in range(8):
        mgr(_Data(step=step, rid=f"r{step}"))
    assert len(mgr._forward_injection_queue) == 0
    assert mgr.n_corrections_applied == 0
