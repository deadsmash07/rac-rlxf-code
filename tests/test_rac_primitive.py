"""RAC primitive end-to-end: apply update, verify bounds."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC.parent))

from src.rac import apply_rac, RACUpdate, exp_age_discount, clipped_is_ratio


def test_rac_produces_bounded_correction():
    """At any realistic Δ, |correction| ≤ (1+ε)·|δ|."""
    A_partial = torch.tensor(0.5)
    slow = torch.tensor(0.7)
    fast_mean = torch.tensor(0.4)
    log_pi_b = torch.tensor(-0.7)
    log_pi_c = torch.tensor(-0.5)  # policy moved slightly toward higher-prob action

    update = apply_rac(
        A_partial=A_partial,
        slow_reward=slow,
        fast_reward_mean=fast_mean,
        log_pi_behavior=log_pi_b,
        log_pi_current=log_pi_c,
        delay=10,
        tau_age=50.0,
        is_clip=0.2,
    )
    # |correction| ≤ w_age · (1 + ε) · |δ| = exp(-10/50) · 1.2 · 0.3 ≈ 0.295
    assert abs(float(update.correction)) < 0.3
    # IS ratio respects clip (allow float32 tolerance)
    assert 0.8 - 1e-4 <= float(update.is_ratio_clipped) <= 1.2 + 1e-4
    # Age weight is deterministic
    assert abs(update.age_weight - exp_age_discount(10, tau_age=50.0)) < 1e-8


def test_rac_zero_correction_when_fast_predicts_slow():
    """Control-variate should produce δ ≈ 0 when slow = m̂_t(fast)."""
    A_partial = torch.tensor(0.5)
    slow = torch.tensor(0.4)
    fast_mean = torch.tensor(0.4)  # slow perfectly predicted

    update = apply_rac(
        A_partial=A_partial,
        slow_reward=slow,
        fast_reward_mean=fast_mean,
        log_pi_behavior=torch.tensor(0.0),
        log_pi_current=torch.tensor(0.0),
        delay=0,
    )
    assert abs(float(update.correction)) < 1e-6
    assert abs(float(update.total_advantage) - 0.5) < 1e-6


def test_age_discount_monotone_decreasing():
    tau = 50.0
    prev = 1.0
    for delta in [0, 10, 25, 50, 100, 200]:
        val = exp_age_discount(delta, tau)
        assert val <= prev
        prev = val
    assert exp_age_discount(0, tau) == 1.0


def test_clipped_is_ratio_vtrace_1sided():
    """V-trace canonical: rho^clip = min(rho_bar, pi_target/pi_behavior).

    Paper Assumption A1 (rho_clip <= rho_bar a.s.) — Proposition 1's TV
    bound depends on this 1-sided clip; a 2-sided PPO-style clamp would
    violate the assumption when the ratio is below the lower bound.
    """
    import torch
    # log ratio = 5 → ratio e^5 ≈ 148; upper-clip to rho_bar = 1.0
    out = clipped_is_ratio(torch.tensor(5.0), torch.tensor(0.0), rho_bar=1.0)
    assert abs(float(out) - 1.0) < 1e-6
    # log ratio = -5 → ratio e^{-5} ≈ 0.0067; no lower clip, passes through
    out = clipped_is_ratio(torch.tensor(-5.0), torch.tensor(0.0), rho_bar=1.0)
    assert abs(float(out) - 0.006737947) < 1e-6
    # Backward-compat: epsilon kwarg maps to rho_bar = 1 + epsilon
    out = clipped_is_ratio(torch.tensor(5.0), torch.tensor(0.0), epsilon=0.2)
    assert abs(float(out) - 1.2) < 1e-6
