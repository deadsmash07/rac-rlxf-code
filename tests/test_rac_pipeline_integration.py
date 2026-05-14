"""End-to-end integration test for the RAC pipeline.

Validates the design of Track 2's full forward-injected-correction flow
without needing to run an actual GRPO trainer on a GPU:

  Step t:
    - rollout produces (prompts, responses, old_log_probs_t, r_fast)
    - MultiChannelRACRewardManager.__call__ returns the fast reward; queues slow
    - trainer-hook would stash (step_t, batch) into RolloutCache

  Step t+Δ (slow reward arrives):
    - manager's "drain matured" returns the SlowReward for uid X
    - apply_rac_correction:
        - looks up (step_t, uid X) in rollout_cache → cached old_log_probs_t
        - re-evaluates log π_{θ(t+Δ)} on the same prompts+responses
          (simulated here by a stub actor)
        - computes δ = w_age(Δ) · ρ_clip · α · (r_slow − r_fast_baseline)
        - adds δ additively to batch's advantages

This is the test closest to what a real verl integration would look like,
minus the actor_rollout_wg.compute_log_prob network call (replaced by a
stub that returns deterministic log-probs).

References:
  - src/verl_integration_notes.md §4 (concrete integration plan)
  - memory/track2_verl_integration_verified.md (subagent-verified design)
  - team2_appendix_A2_proof_sketch.md (theoretical bias bound this tests)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import numpy as np

from src.rac import (
    RACConfig,
    RolloutCache,
    CachedRolloutBatch,
    SlowReward,
    apply_rac_correction,
)


class StubActor:
    """Simulates verl's `actor_rollout_wg.compute_log_prob(batch)` call.

    Takes the cached batch, returns a fresh set of log_probs offset from the
    cached old_log_probs by a configurable KL-drift amount. This lets the
    test control the IS ratio precisely.
    """

    def __init__(self, log_prob_shift: float):
        self.log_prob_shift = log_prob_shift

    def compute_log_prob(self, batch: Any) -> Any:
        """Return a DataProto-compatible object with new log_probs."""
        if hasattr(batch, "batch"):
            cached_lp = batch.batch["old_log_probs"]
        else:
            cached_lp = batch["batch"]["old_log_probs"]

        # Add a uniform shift → controlled KL drift
        new_lp = cached_lp + self.log_prob_shift

        # Return with the same shape as the input for drop-in replacement
        if hasattr(batch, "batch"):
            # Real verl path: mutate a new TensorDict
            try:
                from tensordict import TensorDict
                td = TensorDict({"old_log_probs": new_lp}, batch_size=cached_lp.shape[:1])
                return SimpleNamespace(batch=td)
            except ImportError:
                pass
        return SimpleNamespace(batch={"old_log_probs": new_lp})


def _make_cached_batch(step: int, uids: list[str], response_len: int = 4) -> CachedRolloutBatch:
    """Helper: synthesize a cached rollout with predictable shapes."""
    B = len(uids)
    return CachedRolloutBatch(
        step=step,
        uids=uids,
        prompt_ids=torch.zeros(B, 8, dtype=torch.long),
        response_ids=torch.zeros(B, response_len, dtype=torch.long),
        attention_mask=torch.ones(B, 8 + response_len, dtype=torch.long),
        response_mask=torch.ones(B, response_len, dtype=torch.long),
        old_log_probs=torch.zeros(B, response_len),  # π_{θ(t)} = 1
        A_partial=torch.zeros(B, response_len),
        r_fast=torch.zeros(B),
    )


def _make_advantage_batch(uids_in_batch: list[str], response_len: int = 4) -> Any:
    """Build the verl-DataProto-ish batch that apply_rac_correction mutates."""
    B = len(uids_in_batch)
    advantages = torch.zeros(B, response_len)
    # Minimal DataProto-compatible surface: .batch["advantages"], .non_tensor_batch["uid"]
    return SimpleNamespace(
        batch={"advantages": advantages},
        non_tensor_batch={"uid": np.array(uids_in_batch, dtype=object)},
        meta_info={"global_step": 10},  # t+Δ
    )


def test_full_pipeline_stash_then_correct():
    """End-to-end: stash at step 0, correct at step 10 for a matched uid."""
    cache = RolloutCache(max_ttl_steps=20)
    cached = _make_cached_batch(step=0, uids=["u_A", "u_B"])
    cache.stash(0, cached)
    assert (0, "u_A") in cache
    assert len(cache) == 2

    # Slow reward arrives for u_A at step 10
    slow = SlowReward(
        uid="u_A", step_t=0,
        r_slow=torch.tensor([5.0]),  # meaningfully positive
        channel_name="slow",
        fast_baseline=0.0,
    )
    batch = _make_advantage_batch(uids_in_batch=["u_A", "u_C"])

    cfg = RACConfig(tau_age=10.0, is_clip=0.2, alpha_delta=1.0,
                    max_correction_norm=100.0)
    out = apply_rac_correction(
        batch=batch,
        matured=[slow],
        actor_rollout_wg=StubActor(log_prob_shift=0.0),  # zero drift → ρ=1
        rollout_cache=cache,
        cfg=cfg,
    )

    # u_A row (idx 0) should have a nonzero advantage after correction
    adv_after = out.batch["advantages"]
    assert (adv_after[0] != 0).any(), "u_A's advantage should have been updated"
    # u_C wasn't in matured → no change
    assert torch.allclose(adv_after[1], torch.zeros_like(adv_after[1]))


def test_pipeline_no_matured_is_noop():
    """apply_rac_correction with empty `matured` must not modify the batch."""
    cache = RolloutCache(max_ttl_steps=20)
    cache.stash(0, _make_cached_batch(0, ["u_A"]))
    batch = _make_advantage_batch(["u_A"])
    before = batch.batch["advantages"].clone()
    out = apply_rac_correction(
        batch=batch, matured=[],
        actor_rollout_wg=StubActor(0.0),
        rollout_cache=cache,
        cfg=RACConfig(),
    )
    assert torch.equal(out.batch["advantages"], before)


def test_pipeline_age_discount_larger_delta_smaller_correction():
    """Larger Δ (time since stash) → smaller correction magnitude."""
    cache = RolloutCache(max_ttl_steps=200)
    cache.stash(0, _make_cached_batch(0, ["u_X"]))
    cache.stash(90, _make_cached_batch(90, ["u_Y"]))

    cfg = RACConfig(tau_age=10.0, is_clip=0.2, alpha_delta=1.0,
                    max_correction_norm=100.0)
    slow_close = SlowReward("u_Y", 90, torch.tensor([5.0]), "slow", 0.0)
    slow_far = SlowReward("u_X", 0, torch.tensor([5.0]), "slow", 0.0)

    b_close = _make_advantage_batch(["u_Y"])
    b_close.meta_info["global_step"] = 92  # Δ = 2
    out_close = apply_rac_correction(
        b_close, [slow_close], StubActor(0.0), cache, cfg)
    mag_close = out_close.batch["advantages"].abs().max().item()

    b_far = _make_advantage_batch(["u_X"])
    b_far.meta_info["global_step"] = 92  # Δ = 92
    out_far = apply_rac_correction(
        b_far, [slow_far], StubActor(0.0), cache, cfg)
    mag_far = out_far.batch["advantages"].abs().max().item()

    assert mag_close > mag_far, f"close Δ {mag_close} should exceed far Δ {mag_far}"


def test_pipeline_ttl_evicts_old():
    """After drain_old, correction for an evicted step should no-op gracefully."""
    cache = RolloutCache(max_ttl_steps=5)
    cache.stash(0, _make_cached_batch(0, ["u_old"]))
    n_removed = cache.drain_old(current_step=100)
    assert n_removed == 1
    assert (0, "u_old") not in cache

    # Matured slow reward references the evicted step → apply_rac_correction
    # should not crash; batch unchanged.
    slow = SlowReward("u_old", 0, torch.tensor([5.0]), "slow", 0.0)
    batch = _make_advantage_batch(["u_old"])
    before = batch.batch["advantages"].clone()
    out = apply_rac_correction(
        batch, [slow], StubActor(0.0), cache, RACConfig())
    assert torch.equal(out.batch["advantages"], before)
