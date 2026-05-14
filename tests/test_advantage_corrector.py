"""Tests for forward-injected RAC δ-correction (`src/rac/advantage_corrector.py`).

Covers:
  - Single-rollout δ computation with known inputs
  - w_age monotone decrease
  - ρ_clip respects ε
  - max_correction_norm safety clamp
  - Shape-mismatch error

References: team2_appendix_A2_proof_sketch.md for the mathematical form.
"""
from __future__ import annotations

import pytest
import torch

from src.rac.advantage_corrector import (
    SlowReward,
    RACConfig,
    compute_rac_delta,
)


def _make_slow(r_slow_val: float = 1.0, step_t: int = 0, baseline: float = 0.0):
    return SlowReward(
        uid="u0",
        step_t=step_t,
        r_slow=torch.tensor([r_slow_val]),
        channel_name="slow",
        fast_baseline=baseline,
    )


def test_delta_is_zero_when_r_slow_equals_baseline():
    """δ = 0 when r_slow equals the fast-channel baseline (no new info)."""
    cfg = RACConfig(tau_age=50.0, is_clip=0.2, alpha_delta=1.0)
    slow = _make_slow(r_slow_val=0.5, baseline=0.5)
    log_pi_beh = torch.zeros(4)
    log_pi_tgt = torch.zeros(4)  # identical policies → ρ = 1
    delta = compute_rac_delta(slow, current_step=5, log_pi_behavior=log_pi_beh,
                              log_pi_target=log_pi_tgt, cfg=cfg)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-8)


def test_delta_scales_with_residual_sign():
    """r_slow > baseline ⇒ positive δ; r_slow < baseline ⇒ negative δ."""
    cfg = RACConfig(tau_age=50.0, is_clip=0.2, alpha_delta=1.0)
    log_pi_beh = torch.zeros(4)
    log_pi_tgt = torch.zeros(4)

    d_pos = compute_rac_delta(_make_slow(r_slow_val=0.8, baseline=0.3),
                              current_step=5,
                              log_pi_behavior=log_pi_beh,
                              log_pi_target=log_pi_tgt, cfg=cfg)
    d_neg = compute_rac_delta(_make_slow(r_slow_val=0.1, baseline=0.5),
                              current_step=5,
                              log_pi_behavior=log_pi_beh,
                              log_pi_target=log_pi_tgt, cfg=cfg)
    assert (d_pos > 0).all()
    assert (d_neg < 0).all()


def test_age_discount_monotone_decreasing():
    """Same slow reward, larger Δ → smaller |δ|."""
    cfg = RACConfig(tau_age=10.0, is_clip=0.2, alpha_delta=1.0)
    log_pi_beh = torch.zeros(4)
    log_pi_tgt = torch.zeros(4)
    slow_close = _make_slow(step_t=5, r_slow_val=1.0, baseline=0.0)
    slow_far = _make_slow(step_t=5, r_slow_val=1.0, baseline=0.0)
    d_close = compute_rac_delta(slow_close, current_step=6,  # Δ=1
                                log_pi_behavior=log_pi_beh,
                                log_pi_target=log_pi_tgt, cfg=cfg)
    d_far = compute_rac_delta(slow_far, current_step=50,  # Δ=45
                              log_pi_behavior=log_pi_beh,
                              log_pi_target=log_pi_tgt, cfg=cfg)
    assert d_close.abs().max() > d_far.abs().max()


def test_is_ratio_respects_clip():
    """Large KL drift → ρ clipped to (1+ε); δ bounded by clip · residual · w_age."""
    cfg = RACConfig(tau_age=1000.0, is_clip=0.2, alpha_delta=1.0)  # tau big → w_age≈1
    # Target policy much higher prob than behavior → ρ >> 1, should clip to 1.2
    log_pi_beh = torch.full((4,), -5.0)  # log π_old
    log_pi_tgt = torch.full((4,), -0.1)  # log π_new (ρ = exp(4.9*4) massive)
    slow = _make_slow(step_t=0, r_slow_val=2.0, baseline=0.0)
    delta = compute_rac_delta(slow, current_step=5,
                              log_pi_behavior=log_pi_beh,
                              log_pi_target=log_pi_tgt, cfg=cfg)
    # With ρ_clip=1.2, residual=2.0, w_age ≈ 1.0 → |δ| ≤ ~2.4 per element before clamp
    assert delta.abs().max() <= 1.2 * 2.0 * 1.0 + 1e-4


def test_max_correction_norm_clamps():
    """max_correction_norm clamps runaway δ to cfg.max_correction_norm."""
    cfg = RACConfig(tau_age=1000.0, is_clip=10.0, alpha_delta=100.0,
                    max_correction_norm=3.0)
    slow = _make_slow(step_t=0, r_slow_val=1.0, baseline=0.0)
    log_pi_beh = torch.zeros(4)
    log_pi_tgt = torch.zeros(4)
    delta = compute_rac_delta(slow, current_step=1,
                              log_pi_behavior=log_pi_beh,
                              log_pi_target=log_pi_tgt, cfg=cfg)
    assert delta.abs().max() <= 3.0 + 1e-6


def test_shape_mismatch_raises():
    cfg = RACConfig()
    slow = _make_slow()
    log_pi_beh = torch.zeros(4)
    log_pi_tgt = torch.zeros(8)
    with pytest.raises(ValueError, match="log_pi shape mismatch"):
        compute_rac_delta(slow, current_step=5, log_pi_behavior=log_pi_beh,
                          log_pi_target=log_pi_tgt, cfg=cfg)
