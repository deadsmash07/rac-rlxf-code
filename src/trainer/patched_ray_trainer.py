"""PatchedRayPPOTrainer — verl RayPPOTrainer subclass with RAC hooks.

Design (per memory/track2_verl_integration_verified.md + src/verl_integration_notes.md §4):

verl's `RayPPOTrainer.fit()` (verl/trainer/ppo/ray_trainer.py:1292) computes
advantages at L1552 and consumes them at L1583 via `self._update_actor(batch)`.
Rather than copy-paste the entire fit() method, we override `_update_actor`:

    def _update_actor(self, batch):
        # 1. Drain: fetch any matured slow rewards from reward_manager and
        #    additively inject δ into batch.batch["advantages"] BEFORE actor sees them.
        # 2. Stash: snapshot this step's rollout (old_log_probs, prompt/response
        #    ids, response_mask, advantages, uids) into rollout_cache keyed by
        #    (step, uid), so that a slow reward maturing K steps later can find
        #    the cached behavior-policy log-probs.
        # 3. Delegate to the real actor update.

The order (drain → stash → actor-update) matters: we drain BEFORE stashing the
current step, so we never correct step t with a slow reward that was just
produced at step t (zero lag → no correction needed; cfg.tau_age damps it to
zero anyway, but being explicit prevents edge-case infinite-loops).

Fallback when verl is not importable: `_FALLBACK_BASE` provides a minimal
surface so the subclass can still be unit-tested without a Ray cluster.

References:
  - verl `trainer/ppo/ray_trainer.py:1217` (_update_actor signature)
  - verl `trainer/ppo/ray_trainer.py:1552` (compute_advantage call site)
  - Theorem A3 (forward-injection unbiasedness) in team2_appendix_A3
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from ..rac.advantage_corrector import RACConfig, SlowReward, apply_rac_correction
from ..rac.rollout_cache import CachedRolloutBatch, RolloutCache

try:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer as _VerlRayPPOTrainer
    _HAS_VERL = True
except Exception:  # ImportError OR any environment mis-pin
    _HAS_VERL = False

    class _VerlRayPPOTrainer:  # type: ignore[no-redef]
        """Stub base class for testing without verl installed.

        Exposes the minimal attributes the subclass references
        (actor_rollout_wg, reward_fn, global_steps) so hook tests can
        instantiate without a Ray cluster.
        """

        def __init__(self, *args, **kwargs):
            self.actor_rollout_wg = kwargs.get("actor_rollout_wg", None)
            self.reward_fn = kwargs.get("reward_fn", None)
            self.global_steps = kwargs.get("global_steps", 0)

        def _update_actor(self, batch):
            # No-op placeholder; subclass test overrides via mock
            return batch


def _extract_uids(batch: Any) -> list[str]:
    """Best-effort extraction of per-row uids from a verl DataProto or dict."""
    if hasattr(batch, "non_tensor_batch") and "uid" in batch.non_tensor_batch:
        uid_arr = batch.non_tensor_batch["uid"]
        return [str(u) for u in uid_arr]
    return []


def _snapshot_cached_batch(batch: Any, step: int) -> Optional[CachedRolloutBatch]:
    """Build a CachedRolloutBatch from a verl DataProto-ish `batch`.

    Returns None if the batch is missing any of the essential tensors — the
    caller should treat that as "skip stashing this step" rather than error.
    """
    if not hasattr(batch, "batch"):
        return None
    b = batch.batch
    needed = ["old_log_probs"]
    if any(k not in b for k in needed):
        return None

    old_log_probs = b["old_log_probs"]
    B = old_log_probs.shape[0]
    uids = _extract_uids(batch)
    if len(uids) != B:
        uids = [f"uid_{step}_{i}" for i in range(B)]

    zeros_like_lp = torch.zeros_like(old_log_probs)

    def _get(key, default=None):
        return b.get(key, default)

    return CachedRolloutBatch(
        step=step,
        uids=uids,
        prompt_ids=_get("input_ids", torch.zeros(B, 1, dtype=torch.long)),
        response_ids=_get("responses", torch.zeros(B, 1, dtype=torch.long)),
        attention_mask=_get("attention_mask", torch.ones_like(old_log_probs, dtype=torch.long)),
        response_mask=_get("response_mask", torch.ones_like(old_log_probs, dtype=torch.long)),
        old_log_probs=old_log_probs,
        A_partial=_get("advantages", zeros_like_lp).clone(),
        r_fast=_get("token_level_scores", torch.zeros(B)).detach().clone()
               if _get("token_level_scores") is not None else torch.zeros(B),
    )


class PatchedRayPPOTrainer(_VerlRayPPOTrainer):
    """verl RayPPOTrainer + RAC drain/stash hooks in _update_actor.

    Extra constructor kwargs (beyond what the base class accepts):

        rac_cfg : RACConfig
            Hyperparameters for δ = w_age · ρ_clip · α · (r_slow − r_fast).
        rollout_cache : RolloutCache | None
            If None, one is created with `max_ttl_steps=rac_cfg.tau_age*5`
            (empirical: exp(-5)=0.0067 damps corrections essentially to zero).

    Expectation of the reward manager plugged in as `self.reward_fn`:

        reward_fn.pop_matured() -> list[SlowReward]
            Returns and CLEARS the set of slow-reward records ready to
            correct a past rollout. The MultiChannelRACRewardManager exposes
            this method on top of its existing drain machinery.

    If `reward_fn.pop_matured` is absent, the drain hook is a no-op and this
    class degrades to "RayPPOTrainer + stash only" — useful for profiling
    cache overhead before wiring the slow channel.
    """

    def __init__(self, *args,
                 rac_cfg: Optional[RACConfig] = None,
                 rollout_cache: Optional[RolloutCache] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.rac_cfg = rac_cfg or RACConfig()
        ttl = int(max(self.rac_cfg.tau_age * 5, 10))
        self.rollout_cache = rollout_cache or RolloutCache(max_ttl_steps=ttl)
        self._rac_drained_steps = 0
        self._rac_stashed_rows = 0

        # Tell the reward manager to NOT run its internal _drain_completed —
        # we drive drain here with the correct IS ratio. Double-counting or
        # biased-IS would otherwise follow.
        rm = getattr(self, "reward_fn", None)
        if rm is not None and hasattr(rm, "trainer_managed_drain"):
            rm.trainer_managed_drain = True

    # --- Hooks ---

    def _rac_drain(self, batch: Any) -> Any:
        """Fetch matured slow rewards and inject δ into batch.batch['advantages']."""
        rm = getattr(self, "reward_fn", None)
        if rm is None or not hasattr(rm, "pop_matured"):
            return batch
        matured: list[SlowReward] = rm.pop_matured()
        if not matured:
            return batch
        batch = apply_rac_correction(
            batch=batch,
            matured=matured,
            actor_rollout_wg=self.actor_rollout_wg,
            rollout_cache=self.rollout_cache,
            cfg=self.rac_cfg,
        )
        self._rac_drained_steps += 1
        return batch

    def _rac_stash(self, batch: Any) -> None:
        """Snapshot current step's rollout to rollout_cache."""
        step = int(getattr(batch, "meta_info", {}).get("global_step", self.global_steps))
        cached = _snapshot_cached_batch(batch, step=step)
        if cached is None:
            return
        self.rollout_cache.stash(step, cached)
        self._rac_stashed_rows += len(cached.uids)
        # Opportunistic TTL garbage collection
        self.rollout_cache.drain_old(step)

    # --- Overridden from RayPPOTrainer ---

    def _update_actor(self, batch):
        # Drain BEFORE stashing — so the current step's own stash can't
        # be mis-picked as a "past rollout" to correct (though tau_age
        # damps that case to zero anyway).
        batch = self._rac_drain(batch)
        self._rac_stash(batch)
        return super()._update_actor(batch)
