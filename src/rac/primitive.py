"""Retroactive Advantage Correction primitive.

When a slow reward channel's signal arrives at optimizer-step t+Δ for a
rollout generated at step t, apply:

    A_i^total = A_i^partial(t) + w_age(Δ) · ρ_i^clip · δ_i
    g_i^correct = ρ_i^clip · A_i^total · ∇_θ log π_{θ(t+Δ)}(o_i | q_i)

Per Proposal 2 §3. Cached `log π_{θ(t)}` at rollout time makes ρ_i free.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .importance_sampling import clipped_is_ratio


@dataclass
class RACUpdate:
    """Output of applying RAC to a single pending rollout."""

    rollout_id: str
    original_partial_advantage: torch.Tensor  # A_i^partial(t), pre-correction
    correction: torch.Tensor                  # w_age·ρ_clip·δ
    total_advantage: torch.Tensor             # A_i^total (the updated value)
    is_ratio_clipped: torch.Tensor            # ρ_i^clip
    age_weight: float                         # w_age(Δ)


def apply_rac(
    A_partial: torch.Tensor,       # scalar or (batch,): partial advantage at time t
    slow_reward: torch.Tensor,     # scalar or (batch,): r_slow arrived at t+Δ
    fast_reward_mean: torch.Tensor, # running regression m̂_t(r_fast)
    log_pi_behavior: torch.Tensor, # cached log π_{θ(t)}(o|q) at rollout time
    log_pi_current: torch.Tensor,  # log π_{θ(t+Δ)}(o|q) at correction time
    delay: int,                     # Δ in optimizer steps
    tau_age: float = 50.0,
    is_clip: float = 0.2,           # ε in ρ^clip ∈ [1−ε, 1+ε]
    alpha_delta: float = 1.0,       # scaling on δ
) -> RACUpdate:
    """Apply RAC update to a pending rollout.

    Parameters
    ----------
    A_partial : partial advantage using fast channels only
    slow_reward : the slow reward value that just arrived
    fast_reward_mean : control-variate mean (running regression of slow-on-fast)
    log_pi_behavior : cached log-prob under policy at rollout time
    log_pi_current : log-prob under current policy
    delay : Δ (optimizer steps between rollout and signal arrival)

    Returns
    -------
    RACUpdate with populated correction + total_advantage fields.
    """
    # δ_i = α · (r_slow − m̂_t(r_fast))  (control-variate form)
    delta = alpha_delta * (slow_reward - fast_reward_mean)

    # F-4 FIX (audit): use V-trace 1-sided clip from
    # `importance_sampling.clipped_is_ratio` to match paper Eq (1) and stay
    # consistent with `advantage_corrector.compute_rac_delta`. The previous
    # 2-sided PPO-style clamp `[1-eps, 1+eps]` was inconsistent with the
    # paper's IS semantics and with `clipped_is_ratio` itself (V-trace 1-sided
    # after ).
    rho_clip = clipped_is_ratio(
        log_pi_current, log_pi_behavior, epsilon=is_clip,
    )

    # w_age(Δ)
    import math
    w_age = math.exp(-delay / tau_age)

    correction = w_age * rho_clip * delta
    total = A_partial + correction

    return RACUpdate(
        rollout_id="",  # caller populates
        original_partial_advantage=A_partial.detach().clone(),
        correction=correction.detach().clone(),
        total_advantage=total,
        is_ratio_clipped=rho_clip.detach().clone(),
        age_weight=w_age,
    )
