"""Port of team2_a2_kl_bias_empirical_validation.py — Theorem A2 Pinsker bound.

Verifies ‖A^{π̃} − A^{π_{θ(t+Δ)}}‖_∞ ≤ 2·V_max·(1−γ)^{-1}·√(½·KL) holds across
three KL drift scenarios. The bound is valid but empirically loose by 10-20×.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC.parent))

from src.rac import clipped_is_ratio  # noqa: F401 — keep the import wired
import torch


def softmax(x):
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


def advantage(pi, rewards, gamma=0.9):
    V = np.sum(pi * rewards) / (1 - gamma)
    Q = rewards / (1 - gamma)
    return Q - V


def rac_fixed_point(pi_behavior, pi_target, rho_bar):
    """π̃ ∝ min(ρ̄·π_behavior, π_target)."""
    unnorm = np.minimum(rho_bar * pi_behavior, pi_target)
    return unnorm / unnorm.sum()


@pytest.mark.parametrize(
    "name, pi_behavior, pi_target",
    [
        ("low_KL", np.array([0.5, 0.5]), np.array([0.55, 0.45])),
        ("moderate_KL", np.array([0.5, 0.5]), np.array([0.7, 0.3])),
        ("high_KL", np.array([0.5, 0.5]), np.array([0.9, 0.1])),
    ],
)
def test_pinsker_bound_never_violated(name, pi_behavior, pi_target):
    """Empirical bias must be ≤ Pinsker bound across all ρ̄ values.

    Pinsker bound: ‖A^{π̃} − A^{π_target}‖_∞ ≤ 2·V_max·(1−γ)^{-1}·√(½·KL)
    """
    gamma = 0.9
    V_max = 1.0 / (1 - gamma)
    rewards = np.array([1.0, -1.0])
    kl_drift = kl(pi_target, pi_behavior)
    pinsker_bound = 2 * V_max * (1 / (1 - gamma)) * np.sqrt(0.5 * kl_drift)

    A_target = advantage(pi_target, rewards, gamma)
    for rho_bar in [0.5, 1.0, 2.0, 5.0]:
        pi_tilde = rac_fixed_point(pi_behavior, pi_target, rho_bar)
        A_tilde = advantage(pi_tilde, rewards, gamma)
        empirical_bias = float(np.max(np.abs(A_tilde - A_target)))
        assert empirical_bias <= pinsker_bound, (
            f"{name}, ρ̄={rho_bar}: bias {empirical_bias:.4f} > "
            f"Pinsker bound {pinsker_bound:.4f}"
        )


def test_pinsker_bound_is_loose_as_expected():
    """Under high KL + aggressive truncation, empirical bias should be strictly
    less than the Pinsker bound (bound is not tight)."""
    pi_b = np.array([0.5, 0.5])
    pi_t = np.array([0.9, 0.1])
    gamma = 0.9
    V_max = 1.0 / (1 - gamma)
    rewards = np.array([1.0, -1.0])
    rho_bar = 0.5

    kl_drift = kl(pi_t, pi_b)
    pinsker_bound = 2 * V_max * (1 / (1 - gamma)) * np.sqrt(0.5 * kl_drift)

    pi_tilde = rac_fixed_point(pi_b, pi_t, rho_bar)
    A_tilde = advantage(pi_tilde, rewards, gamma)
    A_target = advantage(pi_t, rewards, gamma)
    bias = float(np.max(np.abs(A_tilde - A_target)))

    # Bound is ~86 here; empirical bias is ~4. Require at least 5× slack.
    assert bias * 5 < pinsker_bound, (
        f"Expected loose bound; bias {bias:.2f}, Pinsker {pinsker_bound:.2f}"
    )
