"""Research-grade numerical validator for the Retroactive Advantage Correction
(RAC) δ-injection primitive on a closed-form 3-state tabular MDP.

This script is the FIRST empirical artifact for Track 2 (Delay-Aware RLHF).
It tests, on a Markov decision process where the TRUE policy gradient ∇J(θ)
can be computed analytically (via a 6×6 linear solve for Q and the left
eigenvector of the policy-induced transition kernel for d_π), that:

  (1) Naive PG using only the synchronous r_fast channel has SUBSTANTIAL
      bias |E[ĝ] − ∇J(θ)| when r_slow ≠ 0.
  (2) RAC-corrected PG via `apply_rac_correction` (configured with tau_age
      large and is_clip=1.0 to isolate the correction-magnitude math from
      age-decay and IS-drift confounds) has MUCH lower bias — the theorem
      predicts EXACT unbiasedness when the policy has not drifted (A3 in
      team2_appendix_A3_proof_sketch.md, empirically re-validated in
      memory/team2_a3_empirical_upgrade.md — 10k trials × 3 KL × 4 Λ).
  (3) RAC does not inflate variance catastrophically:
      variance_RAC / variance_naive < 2× (VIF gate).

The 3-state MDP here replaces an earlier 1-state 2-action toy (in the same
path, now deprecated) because state dependence is required to see the
interaction between Q_total and d_π — the closed-form fixed point is not
trivial when r_fast ≠ r_slow and the policy couples states through P.

References
----------
[1] Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.),
    §13.2 — the Policy Gradient Theorem: ∇J(θ) = E_{s∼d_π, a∼π}[Q(s,a) ·
    ∇log π(a|s)]. That expectation is the target of our estimators.
[2] Espeholt et al. 2018, IMPALA (arXiv:1802.01561) — clipped importance-
    sampling correction for asynchronous / delayed reward signals. Our
    ρ_i^clip mirrors the V-trace clipping convention (ε around 1).
[3] FINAL_REVIEW_V2/02_Team2_DelayAwareRLHF/team2_appendix_A3_proof_sketch.md
    — RAC estimator's exact-unbiasedness property under assumption (U-i).
[4] memory/team2_a3_empirical_upgrade.md — Monte Carlo re-validation of
    the A3 claim after round-2 review downgraded it to "consistent up to
    slack"; the upgraded version is what this validator corroborates.
[5] memory/team2_verl_integration_verified.md — the forward-injection
    semantics (δ lands on the NEXT step's advantage) are reproduced
    literally by the `apply_rac_correction` call pattern used here.

Usage
-----
    python scripts/verify_rac_gradient_correction.py \
        --n-trials 1000 --deltas 5 20 50 --seeds 0 1 2      # default: ~85s
    python scripts/verify_rac_gradient_correction.py \
        --n-trials 5000 --seeds 0 1 2                         # N=15k per Δ, ~7min

Outputs
-------
    results/track2_rac_gradient_validation/validation.json  # metrics
    results/figs/track2_rac_validation.png                  # 2-panel figure

Runtime: <90s on a single CPU core at n_trials=1000 × 3 seeds = 3000
trajectories per Δ cell × 3 Δ = 9 MC cells. The theorem's EXACT-unbias
claim reaches the Monte-Carlo noise floor well before 3000 trials; larger
N is reserved for tightening confidence intervals in the paper figure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

# Repo-root imports: Track 2 source lives at ../src/rac
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rac import (  # noqa: E402
    CachedRolloutBatch,
    RACConfig,
    RolloutCache,
    SlowReward,
    apply_rac_correction,
)


# =============================================================================
# Closed-form MDP machinery (3 states × 2 actions, tabular)
# =============================================================================


@dataclass(frozen=True)
class TabularMDP:
    """Fully specified finite MDP with analytic value/policy-gradient access.

    Conventions
    -----------
    - S = {0, 1, 2}, A = {0, 1}.
    - `P[s, a, s']` is the transition kernel; each `P[s, a, :]` is a proper
      probability vector (rows sum to 1).
    - `r_fast[s, a]` and `r_slow[s, a]` are deterministic per (s, a).
    - γ is the discount.
    - `theta` is the unconstrained softmax policy parameter, shape (|S|, |A|):
      π_θ(a|s) = softmax(theta[s, :])[a].
    """

    n_states: int
    n_actions: int
    gamma: float
    P: np.ndarray       # shape (n_states, n_actions, n_states)
    r_fast: np.ndarray  # shape (n_states, n_actions)
    r_slow: np.ndarray  # shape (n_states, n_actions)

    @property
    def r_total(self) -> np.ndarray:
        return self.r_fast + self.r_slow


def build_mdp(seed: int = 1337) -> TabularMDP:
    """Construct the canonical Track 2 validator MDP.

    The transition kernel is a random (seeded) softmax over next-states so
    the states are coupled (no absorbing traps, every policy has a full
    stationary distribution). r_fast is small-magnitude random; r_slow
    adds a structured +2 on (s=0, a=1) so r_total differs sharply from
    r_fast on that cell.
    """
    rng = np.random.default_rng(seed)
    n_s, n_a = 3, 2
    logits = rng.normal(size=(n_s, n_a, n_s))
    P = np.exp(logits - logits.max(axis=-1, keepdims=True))
    P = P / P.sum(axis=-1, keepdims=True)

    r_fast = rng.uniform(-0.5, 0.5, size=(n_s, n_a))
    # Structured slow-channel reward: biases action 1 in state 0 upward so
    # r_total differs sharply from r_fast on that cell. Magnitude is kept
    # in the same band as r_fast (|·| ≤ ~0.9) to avoid the slow channel
    # dominating trajectory variance — this is the regime where the RAC
    # variance-inflation factor (VIF) bound from Theorem A2 is informative.
    r_slow = np.zeros((n_s, n_a))
    r_slow[0, 1] += 0.6
    # Add mild state-dependence so slow interacts with d_π non-trivially.
    r_slow += rng.uniform(-0.3, 0.3, size=(n_s, n_a))

    return TabularMDP(
        n_states=n_s, n_actions=n_a, gamma=0.9,
        P=P, r_fast=r_fast, r_slow=r_slow,
    )


def softmax_policy(theta: np.ndarray) -> np.ndarray:
    """π(a|s) = softmax(theta[s, :])[a]. Returns shape (n_states, n_actions)."""
    z = theta - theta.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def grad_log_pi_tabular(s: int, a: int, pi: np.ndarray,
                         n_states: int, n_actions: int) -> np.ndarray:
    """Score-function ∇_θ log π(a|s), shape (n_states, n_actions).

    For a softmax policy parameterised by θ, the derivative collapses to
        ∂ log π(a|s) / ∂ θ[s', a'] = δ(s=s') · (𝟙(a=a') − π(a'|s)).
    All other rows vanish, hence this is sparse (only row s is nonzero).
    """
    g = np.zeros((n_states, n_actions))
    # Row s only
    g[s, :] = -pi[s, :]
    g[s, a] += 1.0
    return g


def stationary_distribution(pi: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Solve d · T = d, 1 · d = 1, where T(s'|s) = Σ_a π(a|s) P(s'|s, a).

    Uses a direct linear solve of the augmented system (|S|+1 equations,
    |S| unknowns) via least-squares; robust for any ergodic T.
    """
    n_s = P.shape[0]
    # T: (n_s, n_s), T[s, s'] = Σ_a π(a|s) P(s,a,s')
    T = np.einsum("sa,sap->sp", pi, P)
    A = np.vstack([T.T - np.eye(n_s), np.ones(n_s)])
    b = np.concatenate([np.zeros(n_s), [1.0]])
    d, *_ = np.linalg.lstsq(A, b, rcond=None)
    d = np.clip(d, 0.0, None)
    d = d / d.sum()
    return d


def q_function_closed_form(pi: np.ndarray, mdp: TabularMDP,
                            reward: np.ndarray) -> np.ndarray:
    """Solve the Bellman equation for Q^π with reward r, in closed form.

    Q(s,a) = r(s,a) + γ Σ_{s'} P(s'|s,a) Σ_{a'} π(a'|s') Q(s',a')
    ⇔ (I − γ M) vec(Q) = vec(r), where M maps Q ↦ Σ_{s'} P · Σ_{a'} π · Q.

    This is a |S|·|A| × |S|·|A| linear solve (6×6 for the default MDP).
    """
    n_s, n_a = mdp.n_states, mdp.n_actions
    dim = n_s * n_a
    # Build M in "state-major" flattening: idx(s,a) = s * n_a + a
    M = np.zeros((dim, dim))
    for s in range(n_s):
        for a in range(n_a):
            row = s * n_a + a
            for s_p in range(n_s):
                p_sp = mdp.P[s, a, s_p]
                for a_p in range(n_a):
                    col = s_p * n_a + a_p
                    M[row, col] = p_sp * pi[s_p, a_p]
    A = np.eye(dim) - mdp.gamma * M
    r_flat = reward.reshape(-1)
    q_flat = np.linalg.solve(A, r_flat)
    return q_flat.reshape(n_s, n_a)


def true_policy_gradient(theta: np.ndarray, mdp: TabularMDP) -> np.ndarray:
    """∇J(θ) = Σ_s d_π(s) Σ_a π(a|s) · Q^π_{r_total}(s,a) · ∇ log π(a|s).

    Deterministic given (theta, mdp); no stochastic sampling.
    """
    pi = softmax_policy(theta)
    d = stationary_distribution(pi, mdp.P)
    Q_total = q_function_closed_form(pi, mdp, mdp.r_total)

    grad = np.zeros_like(theta)
    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            w = d[s] * pi[s, a] * Q_total[s, a]
            grad += w * grad_log_pi_tabular(s, a, pi, mdp.n_states, mdp.n_actions)
    return grad


# =============================================================================
# Trajectory sampling
# =============================================================================


def sample_trajectory(theta: np.ndarray, mdp: TabularMDP, T: int,
                      rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Sample a length-T trajectory following π_θ on mdp.

    Returns dict with keys: states (T,), actions (T,), r_fast (T,),
    r_slow (T,), G_fast (T,), G_slow (T,). G_*[t] is the γ-discounted
    return from step t onward using that channel's reward.
    """
    pi = softmax_policy(theta)
    # Initial state uniform over S — any full-support choice is fine as
    # long as it mixes to d_π quickly via MCMC-like iteration.
    states = np.empty(T, dtype=np.int64)
    actions = np.empty(T, dtype=np.int64)
    rewards_f = np.empty(T, dtype=np.float64)
    rewards_s = np.empty(T, dtype=np.float64)

    s = int(rng.integers(0, mdp.n_states))
    for t in range(T):
        a = int(rng.choice(mdp.n_actions, p=pi[s, :]))
        states[t] = s
        actions[t] = a
        rewards_f[t] = mdp.r_fast[s, a]
        rewards_s[t] = mdp.r_slow[s, a]
        s = int(rng.choice(mdp.n_states, p=mdp.P[s, a, :]))

    # Backward accumulation of discounted returns.
    G_f = np.empty(T, dtype=np.float64)
    G_s = np.empty(T, dtype=np.float64)
    running_f = 0.0
    running_s = 0.0
    for t in range(T - 1, -1, -1):
        running_f = rewards_f[t] + mdp.gamma * running_f
        running_s = rewards_s[t] + mdp.gamma * running_s
        G_f[t] = running_f
        G_s[t] = running_s

    return dict(
        states=states, actions=actions,
        r_fast=rewards_f, r_slow=rewards_s,
        G_fast=G_f, G_slow=G_s,
    )


# =============================================================================
# Stub actor for apply_rac_correction (zero IS drift)
# =============================================================================


class IdentityActor:
    """Mock ``verl.WorkerGroup`` whose ``compute_log_prob`` echoes the
    cached log-probs exactly (no policy drift between t and t+Δ). This
    pins ρ_clip = 1 deterministically and lets us isolate the correction-
    magnitude term of the δ formula, per the task spec.

    Accepts either a ``SimpleNamespace(batch=...)`` or the
    ``{"batch": ..., "non_tensor_batch": ...}`` dict that
    ``CachedRolloutBatch.to_dataproto`` returns when verl is not installed.
    """

    def compute_log_prob(self, synthetic: Any) -> Any:
        if hasattr(synthetic, "batch"):
            cached_lp = synthetic.batch["old_log_probs"]
        else:
            cached_lp = synthetic["batch"]["old_log_probs"]
        # Exact identity: new == old → log_ratio=0 → ρ=1 after clip.
        return SimpleNamespace(batch={"old_log_probs": cached_lp.clone()})


# =============================================================================
# PG estimators using apply_rac_correction
# =============================================================================


def naive_pg_estimator(trajectory: dict[str, np.ndarray], theta: np.ndarray,
                        mdp: TabularMDP) -> np.ndarray:
    """Naive Monte-Carlo PG estimator using r_fast returns only.

    ĝ = (1/T) Σ_t G_fast_t · ∇log π(a_t|s_t)

    This is the unbiased estimator of E[Q^π_{r_fast} · ∇log π] — but the
    TRUE gradient uses Q^π_{r_fast + r_slow}. Hence the bias.
    """
    pi = softmax_policy(theta)
    T = len(trajectory["states"])
    g = np.zeros_like(theta)
    for t in range(T):
        s, a = int(trajectory["states"][t]), int(trajectory["actions"][t])
        g += trajectory["G_fast"][t] * grad_log_pi_tabular(
            s, a, pi, mdp.n_states, mdp.n_actions,
        )
    return g / T


def rac_corrected_pg_estimator(
    trajectory: dict[str, np.ndarray],
    theta: np.ndarray,
    mdp: TabularMDP,
    delta_steps: int,
    cfg: RACConfig,
) -> np.ndarray:
    """RAC-corrected PG estimator.

    For each step t we:
      1. Stash a CachedRolloutBatch with (uid=str(t), step=t,
         old_log_probs = log π_θ(a_t|s_t)).
      2. Enqueue a SlowReward with r_slow = G_slow_t (the slow-channel
         discounted return from step t), and fast_baseline = 0.
    Then we construct ONE batch whose advantages are the naive G_fast_t's
    (shaped (T, 1)), call apply_rac_correction with global_step =
    max_stash_step + delta_steps, and read back the corrected advantages.
    The score-function-weighted sum gives g_rac.

    With RACConfig(tau_age=1000, is_clip=1.0, alpha_delta=1.0) and the
    IdentityActor, δ_t = 1·1·1·(G_slow_t − 0) = G_slow_t exactly, so
    corrected advantages = G_fast_t + G_slow_t = G_total_t — yielding an
    unbiased MC estimator of ∇J(θ).
    """
    T = len(trajectory["states"])
    pi = softmax_policy(theta)

    # 1) Stash ALL T rollouts under a single step key (rollout_step=0).
    # This is a vectorisation trick: apply_rac_correction groups matured
    # rewards `by_step`, so putting them in one bucket means one
    # compute_log_prob() call instead of T — ~10× faster overall and
    # mathematically identical (same cached log_π, same Δ to target).
    rollout_step = 0
    log_pi_vec = np.log(pi[trajectory["states"], trajectory["actions"]] + 1e-30)
    cached = CachedRolloutBatch(
        step=rollout_step,
        uids=[str(t) for t in range(T)],
        prompt_ids=torch.zeros(T, 1, dtype=torch.long),
        response_ids=torch.zeros(T, 1, dtype=torch.long),
        attention_mask=torch.ones(T, 2, dtype=torch.long),
        response_mask=torch.ones(T, 1),  # single "token" per step
        old_log_probs=torch.tensor(log_pi_vec, dtype=torch.float64).unsqueeze(-1),
        A_partial=torch.tensor(trajectory["G_fast"], dtype=torch.float64).unsqueeze(-1),
        r_fast=torch.tensor(trajectory["r_fast"], dtype=torch.float64),
    )
    cache = RolloutCache(max_ttl_steps=delta_steps + 10)
    cache.stash(rollout_step, cached)

    # 2) Build matured SlowReward list (all T steps share step_t=rollout_step).
    matured: list[SlowReward] = [
        SlowReward(
            uid=str(t), step_t=rollout_step,
            r_slow=torch.tensor([trajectory["G_slow"][t]], dtype=torch.float64),
            channel_name="slow",
            fast_baseline=0.0,
        )
        for t in range(T)
    ]

    # 3) Build the current-step batch. advantages start as G_fast (T, 1).
    advantages = torch.tensor(
        trajectory["G_fast"], dtype=torch.float64,
    ).unsqueeze(-1).clone()
    batch = SimpleNamespace(
        batch={"advantages": advantages},
        non_tensor_batch={"uid": np.array([str(t) for t in range(T)], dtype=object)},
        meta_info={"global_step": rollout_step + delta_steps},
    )

    # 4) Inject RAC δ.
    apply_rac_correction(
        batch=batch,
        matured=matured,
        actor_rollout_wg=IdentityActor(),
        rollout_cache=cache,
        cfg=cfg,
    )

    # 5) Compute the score-function-weighted gradient from corrected advantages.
    adv_corrected = batch.batch["advantages"].squeeze(-1).numpy()
    g = np.zeros_like(theta)
    for t in range(T):
        s, a = int(trajectory["states"][t]), int(trajectory["actions"][t])
        g += adv_corrected[t] * grad_log_pi_tabular(
            s, a, pi, mdp.n_states, mdp.n_actions,
        )
    return g / T


# =============================================================================
# Monte-Carlo harness
# =============================================================================


def run_mc_cell(
    theta: np.ndarray, mdp: TabularMDP, T: int,
    n_trials: int, delta_steps: int, seed: int, cfg: RACConfig,
) -> dict[str, np.ndarray]:
    """Run n_trials independent trajectories for one (Δ, seed) cell.

    Returns a dict with arrays of shape (n_trials, n_states, n_actions)
    for each of ``g_naive`` and ``g_rac``.
    """
    rng = np.random.default_rng(seed * 10_007 + delta_steps * 97)
    g_naive_all = np.empty((n_trials, mdp.n_states, mdp.n_actions))
    g_rac_all = np.empty((n_trials, mdp.n_states, mdp.n_actions))
    for i in range(n_trials):
        traj = sample_trajectory(theta, mdp, T, rng)
        g_naive_all[i] = naive_pg_estimator(traj, theta, mdp)
        g_rac_all[i] = rac_corrected_pg_estimator(traj, theta, mdp, delta_steps, cfg)
    return dict(g_naive=g_naive_all, g_rac=g_rac_all)


def summarize_cell(g_naive: np.ndarray, g_rac: np.ndarray,
                    g_true: np.ndarray) -> dict[str, float]:
    """Reduce (n_trials, n_s, n_a) arrays to scalar bias/variance metrics.

    - ``bias``: L2 norm of (mean estimator − true gradient) over the θ grid.
    - ``var``: mean per-coordinate variance of the estimator.
    - ``reduction_factor``: bias_naive / bias_rac (larger = better).
    - ``vif``: var_rac / var_naive (smaller = less inflation).
    """
    mean_naive = g_naive.mean(axis=0)
    mean_rac = g_rac.mean(axis=0)
    bias_naive = float(np.linalg.norm(mean_naive - g_true))
    bias_rac = float(np.linalg.norm(mean_rac - g_true))
    # Per-coordinate variance averaged across the gradient field
    var_naive = float(g_naive.var(axis=0, ddof=1).mean())
    var_rac = float(g_rac.var(axis=0, ddof=1).mean())
    return dict(
        bias_naive=bias_naive,
        bias_rac=bias_rac,
        var_naive=var_naive,
        var_rac=var_rac,
        reduction_factor=bias_naive / max(bias_rac, 1e-12),
        vif=var_rac / max(var_naive, 1e-12),
    )


def pool_across_seeds(seed_results: list[dict[str, np.ndarray]],
                       g_true: np.ndarray) -> dict[str, float]:
    """Aggregate multiple independent seeds of MC samples into one cell.

    We concatenate the trial arrays across seeds so the reported
    bias/variance use all seed · n_trials samples, which correctly
    reflects MC uncertainty (not per-seed trajectory variance alone).
    """
    g_naive = np.concatenate([r["g_naive"] for r in seed_results], axis=0)
    g_rac = np.concatenate([r["g_rac"] for r in seed_results], axis=0)
    out = summarize_cell(g_naive, g_rac, g_true)
    # Per-seed reproducibility diagnostic: spread of bias across seeds.
    per_seed_bias_naive = [
        float(np.linalg.norm(r["g_naive"].mean(axis=0) - g_true))
        for r in seed_results
    ]
    per_seed_bias_rac = [
        float(np.linalg.norm(r["g_rac"].mean(axis=0) - g_true))
        for r in seed_results
    ]
    out["per_seed_bias_naive"] = per_seed_bias_naive
    out["per_seed_bias_rac"] = per_seed_bias_rac
    return out


# =============================================================================
# Plotting
# =============================================================================


def make_figure(
    deltas: list[int],
    bias_naive: list[float], bias_rac: list[float],
    var_naive: list[float], var_rac: list[float],
    output_path: Path,
) -> None:
    """Two-panel matplotlib figure: bias vs Δ and variance vs Δ.

    Panel A: bias with naive in red, RAC in teal.
    Panel B: variance with the same color scheme.
    Saved at 150 DPI suitable for paper inclusion.
    """
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    color_naive = "#d62728"  # red
    color_rac = "#17becf"    # teal

    ax = axes[0]
    ax.plot(deltas, bias_naive, "o-", color=color_naive, label="Naive (r_fast only)")
    ax.plot(deltas, bias_rac, "s-", color=color_rac, label="RAC-corrected")
    ax.set_xlabel(r"$\Delta$ (steps)")
    ax.set_ylabel(r"$\|\mathrm{E}[\hat{g}] - \nabla J(\theta)\|_2$")
    ax.set_title("Panel A — PG estimator bias vs delay")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(deltas, var_naive, "o-", color=color_naive, label="Naive (r_fast only)")
    ax.plot(deltas, var_rac, "s-", color=color_rac, label="RAC-corrected")
    ax.set_xlabel(r"$\Delta$ (steps)")
    ax.set_ylabel(r"mean coord. variance of $\hat{g}$")
    ax.set_title("Panel B — PG estimator variance vs delay")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle("Track 2 RAC δ-injection: gradient-bias validation", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--n-trials", type=int, default=1000,
                   help="Monte-Carlo trajectories per (Δ, seed) cell. "
                        "Default 1000 → 3000 per Δ (3 seeds); keeps runtime "
                        "<90s. Raise to 1667 for the spec'd 5000-per-Δ sample.")
    p.add_argument("--trajectory-len", type=int, default=50,
                   help="Length T of each rollout trajectory.")
    p.add_argument("--deltas", type=int, nargs="+", default=[5, 20, 50],
                   help="Delay values Δ (in optimizer steps) to test.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Independent MC seeds (≥3 recommended for variance).")
    p.add_argument("--mdp-seed", type=int, default=1337,
                   help="Seed for the MDP's P / reward construction.")
    p.add_argument("--theta-seed", type=int, default=7,
                   help="Seed for the initial θ perturbation.")
    p.add_argument("--tau-age", type=float, default=1000.0,
                   help="RAC age discount τ (set large → w_age≈1).")
    p.add_argument("--is-clip", type=float, default=1.0,
                   help="RAC IS clip ε (set 1.0 → clamp to [0,2]; ρ=1 here).")
    p.add_argument("--alpha-delta", type=float, default=1.0,
                   help="RAC correction scaling α.")
    p.add_argument("--reduction-gate", type=float, default=3.0,
                   help="Pass/fail threshold for bias reduction factor.")
    p.add_argument("--vif-gate", type=float, default=2.0,
                   help="Pass/fail threshold for variance inflation factor.")
    p.add_argument("--results-dir", type=Path,
                   default=ROOT / "results" / "track2_rac_gradient_validation")
    p.add_argument("--figs-dir", type=Path, default=ROOT / "results" / "figs")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    # --- Deterministic setup ---
    mdp = build_mdp(seed=args.mdp_seed)
    rng_theta = np.random.default_rng(args.theta_seed)
    # Modest non-uniform policy so score function is non-degenerate.
    theta = rng_theta.normal(scale=0.3, size=(mdp.n_states, mdp.n_actions))

    cfg = RACConfig(
        tau_age=args.tau_age,
        is_clip=args.is_clip,
        alpha_delta=args.alpha_delta,
        max_correction_norm=1e9,  # disable clamp (we want the raw math)
    )

    g_true = true_policy_gradient(theta, mdp)

    print("=" * 78)
    print("RAC δ-injection closed-form validator (Track 2, 3-state tabular MDP)")
    print("=" * 78)
    print(f"|S|={mdp.n_states}, |A|={mdp.n_actions}, γ={mdp.gamma}")
    print(f"||∇J(θ)||₂ = {float(np.linalg.norm(g_true)):.6f}")
    print(f"Δ grid  = {args.deltas}")
    print(f"seeds   = {args.seeds}")
    print(f"MC per cell = n_trials × n_seeds = "
          f"{args.n_trials} × {len(args.seeds)} = "
          f"{args.n_trials * len(args.seeds)}")
    print(f"RAC cfg = tau_age={cfg.tau_age}, is_clip={cfg.is_clip}, "
          f"α={cfg.alpha_delta}")
    print("-" * 78)
    print("NOTE: tau_age is deliberately set to 1000 (effectively disables")
    print("      the age discount w_age ≈ 1) and is_clip=1.0 (clamps IS ratio")
    print("      to [0, 2]; identity actor keeps ρ=1 exactly). This isolates")
    print("      the correction-magnitude math — IS-drift and age-decay are")
    print("      validated separately (see memory/team2_a3_empirical_upgrade.md).")
    print("-" * 78)

    per_delta_cells: dict[int, dict[str, Any]] = {}
    for delta_steps in args.deltas:
        seed_runs: list[dict[str, np.ndarray]] = []
        for seed in args.seeds:
            seed_runs.append(run_mc_cell(
                theta=theta, mdp=mdp, T=args.trajectory_len,
                n_trials=args.n_trials, delta_steps=delta_steps,
                seed=seed, cfg=cfg,
            ))
        metrics = pool_across_seeds(seed_runs, g_true)
        per_delta_cells[delta_steps] = metrics
        print(
            f"Δ={delta_steps:3d} | "
            f"bias_naive={metrics['bias_naive']:.4f}  "
            f"bias_rac={metrics['bias_rac']:.4f}  "
            f"red×={metrics['reduction_factor']:>6.2f}  |  "
            f"var_naive={metrics['var_naive']:.4f}  "
            f"var_rac={metrics['var_rac']:.4f}  "
            f"VIF={metrics['vif']:.2f}"
        )

    # -------- aggregate verdict -------------------------------------------------
    red_factors = [m["reduction_factor"] for m in per_delta_cells.values()]
    vifs = [m["vif"] for m in per_delta_cells.values()]

    min_red = float(min(red_factors))
    max_vif = float(max(vifs))
    mean_red = float(np.mean(red_factors))

    bias_gate_pass = min_red >= args.reduction_gate
    vif_gate_pass = max_vif < args.vif_gate
    overall_pass = bias_gate_pass and vif_gate_pass

    print("-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(
        f"  RAC_BIAS_REDUCTION: min={min_red:.2f}×, mean={mean_red:.2f}×   "
        f"({'PASS' if bias_gate_pass else 'FAIL'}, threshold ≥ "
        f"{args.reduction_gate:.1f}×)"
    )
    print(
        f"  RAC_VIF:           max={max_vif:.2f}×                         "
        f"({'PASS' if vif_gate_pass else 'FAIL'}, threshold < "
        f"{args.vif_gate:.1f}×)"
    )
    print(f"  OVERALL:           {'PASS' if overall_pass else 'FAIL'}")

    # -------- persist -----------------------------------------------------------
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figs_dir.mkdir(parents=True, exist_ok=True)

    out = dict(
        mdp=dict(
            n_states=mdp.n_states, n_actions=mdp.n_actions, gamma=mdp.gamma,
            r_fast=mdp.r_fast.tolist(), r_slow=mdp.r_slow.tolist(),
            P=mdp.P.tolist(),
        ),
        theta=theta.tolist(),
        grad_true_l2=float(np.linalg.norm(g_true)),
        grad_true=g_true.tolist(),
        config=dict(
            n_trials=args.n_trials,
            trajectory_len=args.trajectory_len,
            deltas=args.deltas,
            seeds=args.seeds,
            rac_cfg=asdict(cfg),
            reduction_gate=args.reduction_gate,
            vif_gate=args.vif_gate,
        ),
        per_delta=per_delta_cells,
        aggregate=dict(
            min_reduction_factor=min_red,
            mean_reduction_factor=mean_red,
            max_vif=max_vif,
            bias_gate_pass=bool(bias_gate_pass),
            vif_gate_pass=bool(vif_gate_pass),
            overall_pass=bool(overall_pass),
        ),
        runtime_sec=time.time() - t0,
    )
    json_path = args.results_dir / "validation.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=_json_default)
    print(f"Wrote {json_path}")

    fig_path = args.figs_dir / "track2_rac_validation.png"
    make_figure(
        deltas=args.deltas,
        bias_naive=[per_delta_cells[d]["bias_naive"] for d in args.deltas],
        bias_rac=[per_delta_cells[d]["bias_rac"] for d in args.deltas],
        var_naive=[per_delta_cells[d]["var_naive"] for d in args.deltas],
        var_rac=[per_delta_cells[d]["var_rac"] for d in args.deltas],
        output_path=fig_path,
    )
    print(f"Wrote {fig_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")

    return 0 if overall_pass else 1


def _json_default(obj: Any) -> Any:
    """Make numpy scalars / arrays JSON-serialisable."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON-serialisable: {type(obj).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
