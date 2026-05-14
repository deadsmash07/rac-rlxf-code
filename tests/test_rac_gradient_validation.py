"""CI gate: RAC δ-injection bias-reduction on the closed-form tabular MDP.

Pins the headline empirical claim from Track 2 — that RAC's
`apply_rac_correction` primitive yields a policy-gradient estimator whose
bias is substantially lower than the naive-r_fast estimator when the true
reward is r_fast + r_slow.

The full validator (`scripts/verify_rac_gradient_correction.py`) runs for
~60s with n_trials=1000 × 3 seeds. This test is a downscoped version
(n_trials=500, 1 seed, 1 Δ cell) that completes in under 10 seconds and
checks only the reduction-factor gate with a LOOSE threshold (≥ 2.0×)
to tolerate small-sample MC noise. It does NOT check VIF because the
variance estimate is too noisy at n_trials=500 to pin reliably.

Rationale for having this test at all (vs just running the validator):
  - The validator is the published artifact; this test is the CI guard
    that ensures a silent refactor of `apply_rac_correction` cannot
    regress the headline claim without tripping the build.
  - Same test patterns as the other CI pins: 2× threshold is loose
    enough that trajectory randomness within a single seed does not
    cause flakes, but tight enough that an obvious bug (e.g. δ landing
    on the wrong row) still fails fast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Let pytest import the validator module as if it were a library.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import verify_rac_gradient_correction as V  # noqa: E402
from src.rac import RACConfig  # noqa: E402


def test_rac_reduces_pg_bias_on_3state_mdp():
    """Tiny MC: RAC correction must cut PG bias by ≥ 2× at Δ=5, T=50, N=500.

    This is a shrunken version of the full validator:
      - 1 Δ cell (5 steps)
      - 1 seed (0)
      - 500 trajectories (vs 1000 × 3 in production)

    We use the same MDP seed, θ seed, and RAC config as the full run so a
    regression in apply_rac_correction logic causes an immediate failure
    at the same code path the published figure uses.
    """
    mdp = V.build_mdp(seed=1337)
    theta = np.random.default_rng(7).normal(scale=0.3, size=(mdp.n_states, mdp.n_actions))
    cfg = RACConfig(tau_age=1000.0, is_clip=1.0, alpha_delta=1.0,
                    max_correction_norm=1e9)
    g_true = V.true_policy_gradient(theta, mdp)

    cell = V.run_mc_cell(
        theta=theta, mdp=mdp, T=50,
        n_trials=500, delta_steps=5, seed=0, cfg=cfg,
    )
    metrics = V.summarize_cell(cell["g_naive"], cell["g_rac"], g_true)

    # Hard-floor regression gate. In a clean run this number is ~8-13×;
    # 2.0× is where we start calling it "broken".
    assert metrics["reduction_factor"] >= 2.0, (
        f"RAC bias reduction fell below 2.0× "
        f"(reduction_factor={metrics['reduction_factor']:.2f}×, "
        f"bias_naive={metrics['bias_naive']:.4f}, "
        f"bias_rac={metrics['bias_rac']:.4f}). "
        "Run scripts/verify_rac_gradient_correction.py for the full picture."
    )
    # Also check that RAC bias is actually small in absolute terms — bias
    # reduction could be large just because naive blew up; this guards
    # the ‘corrected is correct’ half of the claim.
    assert metrics["bias_rac"] <= 0.1 * float(np.linalg.norm(g_true)) + 1e-3, (
        f"RAC bias is not small relative to ||∇J||: "
        f"bias_rac={metrics['bias_rac']:.4f}, "
        f"||∇J||={float(np.linalg.norm(g_true)):.4f}"
    )


def test_validator_true_gradient_is_deterministic():
    """Sanity guard: `true_policy_gradient` is a pure function of (θ, mdp).

    If someone introduces stochasticity into the closed-form path (e.g.
    swaps lstsq for an iterative solver), the "ground truth" that all
    MC bias numbers compare to would drift silently. This test pins the
    deterministic-return property.
    """
    mdp = V.build_mdp(seed=1337)
    theta = np.random.default_rng(7).normal(scale=0.3, size=(mdp.n_states, mdp.n_actions))
    g1 = V.true_policy_gradient(theta, mdp)
    g2 = V.true_policy_gradient(theta, mdp)
    np.testing.assert_allclose(g1, g2, atol=1e-12)
    # Bellman-solve Q is finite.
    assert np.all(np.isfinite(g1))


def test_rac_estimator_reduces_to_naive_when_slow_is_zero():
    """Corollary: if G_slow ≡ 0 for all rollouts, RAC == naive.

    This is a pure plumbing check on the δ pathway — when SlowReward.r_slow
    is identically zero (same as fast_baseline), compute_rac_delta returns 0,
    so `advantages` after apply_rac_correction equals G_fast exactly, and
    g_rac must equal g_naive. Any discrepancy indicates an ordering or
    shape bug in the injection loop.
    """
    mdp = V.build_mdp(seed=1337)
    theta = np.random.default_rng(7).normal(scale=0.3, size=(mdp.n_states, mdp.n_actions))
    cfg = RACConfig(tau_age=1000.0, is_clip=1.0, alpha_delta=1.0,
                    max_correction_norm=1e9)

    # Force G_slow to zero on every rollout by zero-ing the slow channel.
    mdp_no_slow = V.TabularMDP(
        n_states=mdp.n_states, n_actions=mdp.n_actions, gamma=mdp.gamma,
        P=mdp.P, r_fast=mdp.r_fast,
        r_slow=np.zeros_like(mdp.r_slow),
    )
    rng = np.random.default_rng(42)
    traj = V.sample_trajectory(theta, mdp_no_slow, T=20, rng=rng)
    g_n = V.naive_pg_estimator(traj, theta, mdp_no_slow)
    g_r = V.rac_corrected_pg_estimator(traj, theta, mdp_no_slow, 5, cfg)
    np.testing.assert_allclose(g_n, g_r, atol=1e-8)
