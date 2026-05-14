"""Forward-injected RAC δ-correction applied to the NEXT batch's advantages.

Per memory/track2_verl_integration_verified.md: verl's `RayPPOTrainer.fit()`
consumes advantages IMMEDIATELY (`_update_actor` at ray_trainer.py:1576), so
we CANNOT retroactively mutate a consumed rollout's advantage. Instead:

  1. At rollout time t, stash `(uid, old_log_probs, A_partial, r_fast)` via
     `reward_manager.stash_rollout()`.
  2. When slow reward `r_slow_i` arrives at step t+Δ, the reward manager
     queues a `SlowReward` record with (uid, step_t, r_slow, A_partial).
  3. At the next step's `compute_advantage` call, the trainer calls
     `apply_rac_correction(batch, matured, actor_rollout_wg, cfg)` which
     computes δ_i using the current (θ(t+Δ)) actor's log_probs via
     `actor_rollout_wg.compute_log_prob(synthetic_batch)` AND the cached
     log π_{θ(t)} from the rollout cache.
  4. The resulting δ is added as an additive residual to the new batch's
     advantage tensor.

The IS-ratio analysis in Theorem A2 still holds: ρ_i^clip = clip(π_{θ(t+Δ)} /
π_{θ(t)}, 1-ε, 1+ε) with both log-probs valid at their respective time
points, regardless of which step consumes the correction.

References:
  - verl `verl/trainer/ppo/ray_trainer.py:1552` (compute_advantage site)
  - verl `verl/trainer/ppo/ray_trainer.py:1494` (old_log_prob batch.union site)
  - Theorem A2 in FINAL_REVIEW_V2/02_Team2_DelayAwareRLHF/team2_appendix_A2_proof_sketch.md
  - Espeholt et al. 2018 IMPALA V-trace IS correction (`1802.01561`)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .age_discount import exp_age_discount
from .importance_sampling import clipped_is_ratio


@dataclass
class SlowReward:
    """One matured slow-channel reward ready to correct a past rollout."""

    uid: str
    step_t: int           # optimizer step when the rollout was produced
    r_slow: torch.Tensor  # shape (response_length,) or scalar
    channel_name: str
    fast_baseline: float  # r_fast at rollout time (for control variate)


@dataclass
class RACConfig:
    tau_age: float = 50.0
    is_clip: float = 0.2
    alpha_delta: float = 1.0
    # Upper bound on |δ| after all weighting — prevents runaway single-sample corrections
    max_correction_norm: float = 5.0


def compute_rac_delta(
    matured: SlowReward,
    current_step: int,
    log_pi_behavior: torch.Tensor,  # cached log π_{θ(t)}(a|s) at rollout time
    log_pi_target: torch.Tensor,     # freshly computed log π_{θ(t+Δ)}(a|s)
    cfg: RACConfig,
) -> torch.Tensor:
    """Compute one rollout's δ-correction term.

    δ_i = w_age(Δ) · ρ_i^clip · α · (r_slow_i − m̂_t(r_fast_i))

    Parameters
    ----------
    matured : SlowReward
        The arriving slow-channel reward + cached fast-channel baseline.
    current_step : int
        Current optimizer step t+Δ.
    log_pi_behavior : torch.Tensor
        log π_{θ(t)}(a|s) cached at rollout time (per-token, response_length).
    log_pi_target : torch.Tensor
        log π_{θ(t+Δ)}(a|s) re-evaluated with the CURRENT actor. Must be
        the same shape as `log_pi_behavior`.
    cfg : RACConfig
        Hyperparameters (tau_age, is_clip, alpha_delta, max_correction_norm).

    Returns
    -------
    delta : torch.Tensor
        Shape-matched correction tensor; can be added as an additive residual
        to the current batch's advantage tensor.
    """
    if log_pi_behavior.shape != log_pi_target.shape:
        raise ValueError(
            f"log_pi shape mismatch: behavior {log_pi_behavior.shape} vs "
            f"target {log_pi_target.shape}"
        )

    delta_steps = max(current_step - matured.step_t, 0)
    w_age = exp_age_discount(delta_steps, tau_age=cfg.tau_age)

    # Sequence-level IS ratio via sum-of-log-probs (masks already applied upstream).
    # `clipped_is_ratio` is V-trace 1-sided: min(rho_bar, pi_target/pi_behavior).
    # The legacy `is_clip` (2-sided PPO-style clip-amount eps) is mapped to
    # rho_bar = 1.0 + eps via the deprecated-alias path for backward compat.
    seq_log_pi_target = log_pi_target.sum(dim=-1, keepdim=True)
    seq_log_pi_behavior = log_pi_behavior.sum(dim=-1, keepdim=True)
    rho_clip = clipped_is_ratio(
        seq_log_pi_target, seq_log_pi_behavior, epsilon=cfg.is_clip,
    ).squeeze(-1)

    # Control-variate residual on the slow reward (r_slow − r_fast_baseline)
    # matched to response-length shape for additive residual
    residual = cfg.alpha_delta * (matured.r_slow - matured.fast_baseline)
    if residual.dim() == 0:
        residual = residual.unsqueeze(0)

    delta = w_age * rho_clip * residual

    # Safety: clamp the final norm to prevent one anomalous slow reward from
    # dominating the next step's advantage gradient.
    norm = delta.abs().max()
    if norm > cfg.max_correction_norm:
        delta = delta * (cfg.max_correction_norm / (norm + 1e-8))
    return delta


def apply_rac_correction(
    batch: Any,
    matured: list[SlowReward],
    actor_rollout_wg: Any,
    rollout_cache: Any,
    cfg: RACConfig,
) -> Any:
    """Additively inject the δ-correction into `batch`'s advantage tensor.

    Called by the trainer hook at `ray_trainer.py:1552-ish` just before
    `compute_advantage`. Signature matches the hook planned in
    `src/verl_integration_notes.md` §4.

    Parameters
    ----------
    batch : verl.DataProto
        The current step's batch (already has `advantages` key populated by
        verl's `compute_advantage` if called before us, or empty if after).
    matured : list[SlowReward]
        List of slow-reward records that have completed. Each has a
        `step_t` that points into the `rollout_cache`.
    actor_rollout_wg : verl.WorkerGroup
        The actor worker group. We call `compute_log_prob(synthetic)` to
        re-evaluate log π_{θ(t+Δ)} on the cached rollout's prompts+responses.
    rollout_cache : RolloutCache
        Ring buffer keyed by (step, uid) → {old_log_probs, response_mask,
        A_partial, r_fast}.
    cfg : RACConfig

    Returns
    -------
    batch : verl.DataProto
        The same batch with `batch.batch["advantages"]` updated additively.
    """
    if not matured:
        return batch

    # Group matured by step_t so we minimize recomputes of log π_{θ(t+Δ)}.
    from collections import defaultdict
    by_step: dict[int, list[SlowReward]] = defaultdict(list)
    for s in matured:
        by_step[s.step_t].append(s)

    deltas_by_uid: dict[str, torch.Tensor] = {}
    current_step = int(getattr(batch, "meta_info", {}).get("global_step", 0) or 0)

    for step_t, group in by_step.items():
        # Fetch cached rollouts
        cached = rollout_cache.get_batch(step_t, [s.uid for s in group])
        if cached is None or len(cached.uids) == 0:
            continue
        # Re-evaluate log π_{θ(t+Δ)} on the cached prompts+responses.
        # In verl this uses `actor_rollout_wg.compute_log_prob(synthetic_batch)`.
        synthetic = cached.to_dataproto()
        log_pi_target = actor_rollout_wg.compute_log_prob(synthetic)
        # `compute_log_prob` returns a DataProto with `old_log_probs` by convention.
        log_pi_target_tensor = log_pi_target.batch.get("old_log_probs")
        if log_pi_target_tensor is None:
            log_pi_target_tensor = log_pi_target.batch.get("log_probs")

        # F-3 FIX (audit): use uid->row-index dict from
        # `cached.uids` instead of positional enumerate over `group`. When some
        # requested uids are missing from the cache, `cached.uids` is a subset
        # and positional alignment between `group` and `cached.*` is wrong —
        # the wrong rollout's log-probs would be applied.
        uid_to_row = {uid: idx for idx, uid in enumerate(cached.uids)}
        for slow in group:
            row_i = uid_to_row.get(slow.uid, None)
            if row_i is None:
                # Cached batch did not include this uid; skip.
                continue
            log_pi_behavior_i = cached.old_log_probs[row_i]
            log_pi_target_i = log_pi_target_tensor[row_i]
            # Mask out padding positions
            mask_i = cached.response_mask[row_i]
            log_pi_behavior_i = log_pi_behavior_i * mask_i
            log_pi_target_i = log_pi_target_i * mask_i

            delta = compute_rac_delta(
                matured=slow,
                current_step=current_step,
                log_pi_behavior=log_pi_behavior_i,
                log_pi_target=log_pi_target_i,
                cfg=cfg,
            )
            deltas_by_uid[slow.uid] = delta

    # Additively inject into current batch's advantages.
    # This requires the batch to know which uids correspond to which rows.
    # If verl attaches uid to non_tensor_batch, we iterate; otherwise this is
    # a no-op (the corrections are added only on matching uids).
    if not hasattr(batch, "batch") or "advantages" not in batch.batch:
        return batch
    current_uids = batch.non_tensor_batch.get("uid", [])
    for row_idx, uid in enumerate(current_uids):
        if uid in deltas_by_uid:
            batch.batch["advantages"][row_idx] += deltas_by_uid[uid]
    return batch
