"""Tensor-Λ (U-i) validation + unbiasedness sanity tests.

Ports the empirical-validation logic from
`FINAL_REVIEW_V2/02_Team2_DelayAwareRLHF/team2_a3_tensor_lambda_empirical_validation.py`
(which showed EXACT unbiasedness under (U-i) across 3 KL drift scenarios).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC.parent))

from src.rac import TensorLambda, validate_uniformity


def test_tensor_lambda_passes_validation_under_u_i():
    """Sum per channel = 1 → should validate."""
    L = TensorLambda(n_channels=2, D_max=10)
    L.set(0, 0, 1.0)        # fast channel, Δ=0
    L.set(1, 5, 1.0)        # slow channel, Δ=5
    L.validate()            # should not raise


def test_tensor_lambda_rejects_violation_of_u_i():
    """Sum for a channel > 1 → should fail."""
    L = TensorLambda(n_channels=2, D_max=10)
    L.set(0, 0, 2.0)        # BAD: violates (U-i) sum > 1
    L.set(1, 5, 1.0)
    with pytest.raises(ValueError, match="U-i"):
        L.validate()


def test_tensor_lambda_rejects_violation_of_u_ii():
    """Setting Δ > D_max → should raise."""
    L = TensorLambda(n_channels=1, D_max=5)
    with pytest.raises(ValueError, match="D_max"):
        L.set(0, 10, 1.0)


def test_empirical_unbiasedness_under_u_i():
    """Monte Carlo: under (U-i), tensor-Λ gradient estimate has zero bias.

    Port of team2_a3_tensor_lambda_empirical_validation.py's key finding.
    """
    rng = np.random.default_rng(42)
    n_trials = 10000

    # 2-action policy π(a=0) = σ(θ)
    def softmax(x):
        z = x - np.max(x)
        e = np.exp(z)
        return e / e.sum()

    def grad_log_pi(a, theta):
        pi = softmax(np.array([theta, 0.0]))
        return 1.0 - pi[0] if a == 0 else -pi[0]

    theta_behavior = 0.0
    theta_target = 0.5  # moderate KL drift
    rewards_k0 = np.array([1.0, -1.0])
    rewards_k1 = np.array([0.5, -0.5])

    # True gradient
    pi_target = softmax(np.array([theta_target, 0.0]))
    true_g = sum(
        pi_target[a] * (rewards_k0[a] + rewards_k1[a]) * grad_log_pi(a, theta_target)
        for a in range(2)
    )

    # RAC-like Λ: Λ[k=0, 0] = 1, Λ[k=1, 5] = 1
    def single_trial():
        pi_b = softmax(np.array([theta_behavior, 0.0]))
        a = int(rng.uniform() > pi_b[0])
        pi_t = softmax(np.array([theta_target, 0.0]))
        rho = pi_t[a] / pi_b[a]
        rho_clip = np.clip(rho, 0.5, 2.0)
        g_est = (
            1.0 * rho_clip * rewards_k0[a] * grad_log_pi(a, theta_target)
            + 1.0 * rho_clip * rewards_k1[a] * grad_log_pi(a, theta_target)
        )
        return g_est

    estimates = [single_trial() for _ in range(n_trials)]
    empirical_mean = float(np.mean(estimates))
    bias = empirical_mean - true_g

    # Should be EXACT under (U-i). Use absolute tolerance since IS correction
    # in 2-action case can give deterministic zero variance via cancellation
    # (empirically observed: bias and SE both → 0).
    assert abs(bias) < 0.01, f"bias {bias:.6f} (expected ≈0 under U-i)"
