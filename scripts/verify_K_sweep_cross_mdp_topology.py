""" §4.5 cross-MDP-topology K-sweep replication driver.

Pre-registered in `PREREG_T2_K_SWEEP_CROSS_MDP_TOPOLOGY.md` (commit 453363a,
2026-04-26 23:08 IST). Implements the design verbatim — 5 hand-designed
small-tabular MDP topologies x 7 K values x 5 MDP seeds x 3 MC seeds x 3000
trajectories = 525 cells x 3000 trajectories each = 1.575M trajectories.

THE SCIENTIFIC QUESTION (addresses reviewer concern):

  /185/186 K-sweep on a SINGLE 3-state x 2-action MDP gives
  reduction-factor curve K in {2,3,5,7,10,15,20} -> {47.9, 22.65, 32.95,
  39.8, 48.3, 57.48 (PEAK), 54.83 (plateau)}. Reviewer attack: "K=15 peak
  is MDP-specific not algorithmic". This driver tests whether the K-curve
  qualitative shape and K-peak location generalises across alternative
  small tabular MDP topologies (varying state-count, action-count,
  transition density, reward sparsity).

Topology specifications (verbatim from PREREG sec 2.2):

  T0 canonical_3s2a    : reuses V2.build_mdp(seed) verbatim
                         (3 states, 2 actions, dense softmax-of-Gaussian P,
                          structured r_slow +0.6 on (s=0,a=1))
  T1 chain_5s2a        : 5-state chain, action 0 stays + drift left,
                         action 1 moves right; r_slow concentrated on
                         (s=4, a=1) terminal-like attractor
  T2 cyclic_4s3a       : 4-state cyclic, action a deterministically
                         advances state by a (mod 4) with 0.85 prob; dense
                         r_slow uniform U(-0.3, 0.3)
  T3 dense_5s3a        : 5-state x 3-action analog of T0 with full-support
                         random transitions; r_slow +0.6 on (s=0, a=2)
  T4 terminal_3s2a     : 3-state x 2-action with absorbing terminal at s=2;
                         r_slow concentrated on (s=0, a=0) attracting AWAY
                         from terminal

K range and per-channel deltas (from PREREG sec 2.1, 2.4):

  K=2: anchor case. Uses V2.naive_pg_estimator + V2.rac_corrected_pg_estimator
       at delta_steps=20 (matches  /  47.9x headline at
       tau_age=200 if we wanted to hit it; here we hold tau_age=1000 to
       match the K>=3 RAC config and isolate the correction-magnitude math).
       2-channel decomposition: (r_fast, r_slow) verbatim per topology.

  K in {3,5,7,10,15,20}: per-channel decomposition r_true / K + zero-sum
       noise; deltas Delta_k = k for k=1..K-1; Delta_fast=0.

PREREG branches (4-way mutually exclusive; FALSIFIED checked first):

  Branch 4 FALSIFIED-K-PEAK-NON-GENERALISED:
    ANY topology has peak_K in {2,3,5} OR
    ANY topology has min_K reduction <= 1.5 OR
    >=2 topologies have peak_band excluding all of {10,15,20}

  Branch 1 ESTABLISHED-K-PEAK-CROSS-TOPOLOGY:
    ALL 5 topologies have peak_band including K in {10,15,20} AND
    ALL 5 topologies have min_K reduction >= 5 AND
    >=4 of 5 topologies have peak_band including K=15 specifically AND
    ALL 5 topologies have max VIF-fast across K < 2.0

  Branch 2 MODERATE-K-PEAK-ROBUST:
    >=3 of 5 topologies have peak_band including K=15 AND
    ALL 5 topologies have min_K reduction >= 3 AND
    NO topology has peak shifted LEFT to {2,3,5}

  Branch 3 PRELIMINARY-K-PEAK-TOPOLOGY-DEPENDENT:
    >=1 alternative topology has peak_band including K=15 AND
    Branches 1/2/4 fail.

Reuses (verbatim, no modification):
  src.rac.{RACConfig, RolloutCache, SlowReward, TensorLambda,
           apply_rac_correction, CachedRolloutBatch}
  verify_rac_gradient_correction.{IdentityActor, softmax_policy,
                                   grad_log_pi_tabular,
                                   stationary_distribution,
                                   q_function_closed_form,
                                   true_policy_gradient,
                                   sample_trajectory,
                                   naive_pg_estimator,
                                   rac_corrected_pg_estimator,
                                   _json_default}
  verify_tensor_lambda_multichannel_K10.{TabularMDPK,
                                          sample_trajectory_k,
                                          naive_fast_only_pg,
                                          partial_fast_half_pg,
                                          tensor_lambda_rac_pg_k,
                                          per_channel_rac_pg_k,
                                          make_lambda_uniform_k,
                                          summarize_cell,
                                          pool_across_seeds}

NEW logic (this script only):
  - 5 topology factories: build_mdp_<name>(seed)
  - K-channel decomposition that accepts an arbitrary V2.TabularMDP base
    (analog of build_mdp_kchannel from K10 driver but topology-agnostic)
  - Outer driver loops topology x K x mdp_seed x mc_seed
  - Per-(topology, K) summary: reduction_topology_mean/min/max,
    peak_K_topology, peak_band_topology
  - Verdict logic verbatim from PREREG sec 4

Outputs:
  results/track2_K_sweep_cross_mdp_topology/summary.json    # full sweep
  results/track2_K_sweep_cross_mdp_topology/topology_<T>.json   # per-topology
  results/figs/track2_K_sweep_cross_mdp_topology.png           # 4-panel figure

Wall clock estimate (HONEST): per RAC delay test 945.4s for 75 cells x 15 reps x 3000
trials at K=2 only. Here: 525 cells x 15 reps x 3000 trials but mostly
K>=3 which has K-1 channel overhead per trajectory. Estimated 30 min - 3
hr single-thread CPU; PREREG budgets up to 7 hr.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch

# Reuse /170/185/186 infrastructure verbatim.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import verify_rac_gradient_correction as V2  # noqa: E402
from src.rac import (  # noqa: E402
    RACConfig,
    TensorLambda,
)
from verify_rac_gradient_correction import (  # noqa: E402
    TabularMDP,
    softmax_policy,
    grad_log_pi_tabular,
    true_policy_gradient,
    sample_trajectory,
    naive_pg_estimator,
    rac_corrected_pg_estimator,
    _json_default,
)
from verify_tensor_lambda_multichannel_K10 import (  # noqa: E402
    TabularMDPK,
    sample_trajectory_k,
    naive_fast_only_pg,
    partial_fast_half_pg,
    tensor_lambda_rac_pg_k,
    per_channel_rac_pg_k,
    make_lambda_uniform_k,
    summarize_cell as summarize_cell_k,
    pool_across_seeds as pool_across_seeds_k,
)


# =============================================================================
# Topology factories — PREREG sec 2.2
# =============================================================================


def build_mdp_canonical_3s2a(seed: int = 1337) -> TabularMDP:
    """T0: canonical 3-state x 2-action — reuses V2.build_mdp verbatim.

    This is the  /  /  reference topology.
    Reproduces the K-sweep peak-at-K=15 finding when run as the only
    topology in the sweep.
    """
    return V2.build_mdp(seed=seed)


def build_mdp_chain_5s2a(seed: int = 1337) -> TabularMDP:
    """T1: chain 5-state x 2-action with terminal-like attractor.

    Action 0 = stay-with-drift-left (0.7 self-loop, 0.25 to s-1, 0.05 elsewhere).
    Action 1 = move-right (0.7 to s+1, 0.25 self-loop, 0.05 elsewhere).
    r_slow concentrated on (s=4, a=1) (terminal-like reward attractor).
    r_fast small random U(-0.4, 0.4).
    """
    rng = np.random.default_rng(seed)
    n_s, n_a = 5, 2
    P = np.zeros((n_s, n_a, n_s))
    for s in range(n_s):
        # Action 0: stay-or-drift-left
        P[s, 0, s] = 0.7
        if s > 0:
            P[s, 0, s - 1] = 0.25
        else:
            P[s, 0, s] += 0.25  # boundary: can't go left, stay
        residual = 1.0 - P[s, 0, :].sum()
        # Distribute residual uniformly to remaining states
        remaining = [s_p for s_p in range(n_s) if P[s, 0, s_p] == 0]
        if remaining and residual > 0:
            for s_p in remaining:
                P[s, 0, s_p] = residual / len(remaining)
        # Action 1: move-right
        if s < n_s - 1:
            P[s, 1, s + 1] = 0.7
        else:
            P[s, 1, s] = 0.7  # boundary
        P[s, 1, s] += 0.25
        residual = 1.0 - P[s, 1, :].sum()
        remaining = [s_p for s_p in range(n_s) if P[s, 1, s_p] == 0]
        if remaining and residual > 0:
            for s_p in remaining:
                P[s, 1, s_p] = residual / len(remaining)
    # Renormalize defensively
    P = P / P.sum(axis=-1, keepdims=True)

    r_fast = rng.uniform(-0.4, 0.4, size=(n_s, n_a))
    r_slow = np.zeros((n_s, n_a))
    r_slow[4, 1] += 0.6  # terminal-like attractor
    r_slow += rng.uniform(-0.2, 0.2, size=(n_s, n_a))

    return TabularMDP(
        n_states=n_s, n_actions=n_a, gamma=0.9,
        P=P, r_fast=r_fast, r_slow=r_slow,
    )


def build_mdp_cyclic_4s3a(seed: int = 1337) -> TabularMDP:
    """T2: cyclic 4-state x 3-action, dense r_slow.

    Action a deterministically advances state by a (mod 4) with prob 0.85,
    uniform 0.15 to the other 3 states. Dense r_slow uniform U(-0.3, 0.3).
    r_fast small random U(-0.4, 0.4).
    """
    rng = np.random.default_rng(seed)
    n_s, n_a = 4, 3
    P = np.zeros((n_s, n_a, n_s))
    for s in range(n_s):
        for a in range(n_a):
            target = (s + a) % n_s
            P[s, a, target] = 0.85
            others = [s_p for s_p in range(n_s) if s_p != target]
            for s_p in others:
                P[s, a, s_p] = 0.15 / len(others)

    r_fast = rng.uniform(-0.4, 0.4, size=(n_s, n_a))
    r_slow = rng.uniform(-0.3, 0.3, size=(n_s, n_a))  # DENSE, not sparse

    return TabularMDP(
        n_states=n_s, n_actions=n_a, gamma=0.9,
        P=P, r_fast=r_fast, r_slow=r_slow,
    )


def build_mdp_dense_5s3a(seed: int = 1337) -> TabularMDP:
    """T3: dense 5-state x 3-action softmax-of-Gaussian (analog of T0 scaled up).

    Full-support random softmax-of-Gaussian P (no zero-prob transitions);
    r_fast random U(-0.5, 0.5); r_slow structured +0.6 on (s=0, a=2)
    (analog of canonical's (0,1) attractor).
    """
    rng = np.random.default_rng(seed)
    n_s, n_a = 5, 3
    logits = rng.normal(size=(n_s, n_a, n_s))
    P = np.exp(logits - logits.max(axis=-1, keepdims=True))
    P = P / P.sum(axis=-1, keepdims=True)

    r_fast = rng.uniform(-0.5, 0.5, size=(n_s, n_a))
    r_slow = np.zeros((n_s, n_a))
    r_slow[0, 2] += 0.6  # structured attractor analog
    r_slow += rng.uniform(-0.3, 0.3, size=(n_s, n_a))

    return TabularMDP(
        n_states=n_s, n_actions=n_a, gamma=0.9,
        P=P, r_fast=r_fast, r_slow=r_slow,
    )


def build_mdp_terminal_3s2a(seed: int = 1337) -> TabularMDP:
    """T4: 3-state x 2-action with absorbing terminal at s=2.

    P[2, a, :] = e_2 for all a (absorbing). r_slow concentrated on (s=0, a=0)
    attracting AWAY from terminal (so optimal policy tries to avoid getting
    trapped). r_fast small random U(-0.5, 0.5).
    """
    rng = np.random.default_rng(seed)
    n_s, n_a = 3, 2
    logits = rng.normal(size=(n_s, n_a, n_s))
    P = np.exp(logits - logits.max(axis=-1, keepdims=True))
    P = P / P.sum(axis=-1, keepdims=True)
    # Override s=2 to be absorbing
    for a in range(n_a):
        P[2, a, :] = 0.0
        P[2, a, 2] = 1.0

    r_fast = rng.uniform(-0.5, 0.5, size=(n_s, n_a))
    r_slow = np.zeros((n_s, n_a))
    r_slow[0, 0] += 0.6  # attract away from terminal
    r_slow += rng.uniform(-0.3, 0.3, size=(n_s, n_a))

    return TabularMDP(
        n_states=n_s, n_actions=n_a, gamma=0.9,
        P=P, r_fast=r_fast, r_slow=r_slow,
    )


# =============================================================================
# K-channel decomposition that accepts an arbitrary base TabularMDP
# (generalisation of K10 build_mdp_kchannel which hardcoded V2.build_mdp(seed))
# =============================================================================


def decompose_to_k_channels(
    base_mdp: TabularMDP, n_channels: int,
    decomposition_seed: int = 0,
    noise_scale: float = 0.3,
) -> TabularMDPK:
    """Decompose the base MDP's r_total = r_fast + r_slow into K channels.

    Each channel gets (1/K) * r_total + zero-mean noise. The last channel's
    noise is set to the negative sum of the prior K-1 noises so the channel
    sum equals r_total EXACTLY at every (s,a) cell — preserving the
    closed-form Bellman ground-truth invariant.

    decomposition_seed parameterises the noise draw; topology and mdp_seed
    are FIXED via base_mdp.
    """
    r_total = base_mdp.r_fast + base_mdp.r_slow
    rng = np.random.default_rng(decomposition_seed)
    n_s, n_a = base_mdp.n_states, base_mdp.n_actions

    if n_channels < 2:
        raise ValueError(f"Need n_channels >= 2; got {n_channels}.")

    noises_indep = [
        rng.normal(scale=noise_scale, size=(n_s, n_a))
        for _ in range(n_channels - 1)
    ]
    noise_last = -sum(noises_indep)
    noises = noises_indep + [noise_last]

    r_channels = tuple(
        r_total / n_channels + noises[k] for k in range(n_channels)
    )
    # Sanity: channel sum must equal r_total at every cell.
    assert np.allclose(sum(r_channels), r_total, atol=1e-12)

    return TabularMDPK(
        n_states=n_s, n_actions=n_a, gamma=base_mdp.gamma,
        P=base_mdp.P, r_channels=r_channels,
    )


# =============================================================================
# K=2 special case — uses the V2 (fast+slow) machinery, NOT the K-channel one
# =============================================================================


def run_K2_cell(
    base_mdp: TabularMDP, theta: np.ndarray, T: int, n_trials: int,
    delta_steps: int, seed: int, cfg: RACConfig,
) -> dict[str, float]:
    """Reduction-factor cell for K=2 (anchor) using V2 machinery.

    This is verbatim the  K=2 protocol: naive PG with G_fast only
    vs RAC-corrected PG with the slow-channel matured at delta_steps. Returns
    bias_naive, bias_rac, reduction_factor, vif. Pooled across `n_trials`
    trajectories (single seed; outer driver pools across mc_seeds).
    """
    rng = np.random.default_rng(seed * 10_007 + delta_steps * 97)
    g_naive_all = np.empty((n_trials, base_mdp.n_states, base_mdp.n_actions))
    g_rac_all = np.empty((n_trials, base_mdp.n_states, base_mdp.n_actions))
    for i in range(n_trials):
        traj = sample_trajectory(theta, base_mdp, T, rng)
        g_naive_all[i] = naive_pg_estimator(traj, theta, base_mdp)
        g_rac_all[i] = rac_corrected_pg_estimator(
            traj, theta, base_mdp, delta_steps, cfg,
        )
    g_true = true_policy_gradient(theta, base_mdp)
    mean_naive = g_naive_all.mean(axis=0)
    mean_rac = g_rac_all.mean(axis=0)
    bias_naive = float(np.linalg.norm(mean_naive - g_true))
    bias_rac = float(np.linalg.norm(mean_rac - g_true))
    var_naive = float(g_naive_all.var(axis=0, ddof=1).mean())
    var_rac = float(g_rac_all.var(axis=0, ddof=1).mean())
    return dict(
        bias_naive=bias_naive,
        bias_rac=bias_rac,
        var_naive=var_naive,
        var_rac=var_rac,
        reduction_factor=bias_naive / max(bias_rac, 1e-12),
        vif_fast=var_rac / max(var_naive, 1e-12),
    )


def aggregate_K2_per_mdp(
    base_mdp: TabularMDP, theta: np.ndarray, T: int, n_trials: int,
    delta_steps: int, mc_seeds: list[int], cfg: RACConfig,
) -> dict[str, float]:
    """Pool K=2 cell across mc_seeds. Returns per-(topology, mdp_seed)
    aggregate matching the K>=3 schema.
    """
    seed_runs = []
    for s in mc_seeds:
        seed_runs.append(run_K2_cell(
            base_mdp=base_mdp, theta=theta, T=T,
            n_trials=n_trials, delta_steps=delta_steps,
            seed=s, cfg=cfg,
        ))
    # Pool across seeds (concatenate would require keeping arrays; we
    # average reduction factors instead since they're already pooled per-seed)
    red_factors = [r["reduction_factor"] for r in seed_runs]
    vifs = [r["vif_fast"] for r in seed_runs]
    return dict(
        reduction_factor=float(np.mean(red_factors)),
        reduction_min=float(np.min(red_factors)),
        reduction_max=float(np.max(red_factors)),
        vif_fast_mean=float(np.mean(vifs)),
        vif_fast_max=float(np.max(vifs)),
        per_seed_reduction=red_factors,
        per_seed_vif=vifs,
    )


# =============================================================================
# K >= 3 cell driver — uses K10 machinery generalised to any base MDP
# =============================================================================


def run_Kgeq3_cell(
    base_mdp: TabularMDP, theta: np.ndarray, K: int, T: int, n_trials: int,
    deltas: list[int], mc_seeds: list[int], cfg: RACConfig,
    decomposition_seed: int = 0, noise_scale: float = 0.3,
) -> dict[str, Any]:
    """Per-(topology, K, mdp_seed) cell pooled across mc_seeds.

    Uses the K-channel decomposition + tensor-Lambda RAC machinery from
    verify_tensor_lambda_multichannel_K10.py verbatim, with the base_mdp
    swapped from V2.build_mdp(seed) to the topology-specific factory output.
    """
    mdp_k = decompose_to_k_channels(
        base_mdp=base_mdp, n_channels=K,
        decomposition_seed=decomposition_seed, noise_scale=noise_scale,
    )
    g_true = V2.true_policy_gradient(
        theta,
        # Pack r_total into (r_fast=r_total, r_slow=0) for V2.true_policy_gradient
        TabularMDP(
            n_states=mdp_k.n_states, n_actions=mdp_k.n_actions,
            gamma=mdp_k.gamma, P=mdp_k.P,
            r_fast=mdp_k.r_total, r_slow=np.zeros_like(mdp_k.r_total),
        ),
    )
    D_max = max(deltas) + 5
    Lambda, delta_map = make_lambda_uniform_k(deltas=deltas, D_max=D_max)

    # K10 driver's run_mc_cell expects (theta, mdp, T, n_trials, delta_map,
    # Lambda, seed, cfg). We replicate inline to keep per-trajectory loop
    # identical to K10's machinery and to enable per-seed pooling.
    seed_runs: list[dict[str, np.ndarray]] = []
    for s in mc_seeds:
        rng = np.random.default_rng(s * 10_007 + 31 + K * 977)
        delta_per_channel = {
            k: delta_map.get(k, [(1, 1.0)])[0][0] for k in range(1, K)
        }
        gs_fast = np.empty((n_trials, mdp_k.n_states, mdp_k.n_actions))
        gs_half = np.empty((n_trials, mdp_k.n_states, mdp_k.n_actions))
        gs_tL = np.empty((n_trials, mdp_k.n_states, mdp_k.n_actions))
        gs_indep = [
            np.empty((n_trials, mdp_k.n_states, mdp_k.n_actions))
            for _ in range(K - 1)
        ]
        for i in range(n_trials):
            traj = sample_trajectory_k(theta, mdp_k, T, rng)
            gs_fast[i] = naive_fast_only_pg(traj, theta, mdp_k)
            gs_half[i] = partial_fast_half_pg(traj, theta, mdp_k, k_max=2)
            gs_tL[i] = tensor_lambda_rac_pg_k(
                traj, theta, mdp_k, Lambda, delta_map, cfg,
            )
            per_channel = per_channel_rac_pg_k(
                traj, theta, mdp_k, delta_per_channel, cfg,
            )
            for j, g_j in enumerate(per_channel):
                gs_indep[j][i] = g_j
        cell_samples = dict(g_fast=gs_fast, g_half=gs_half, g_tL=gs_tL)
        for j, g_arr in enumerate(gs_indep):
            cell_samples[f"g_indep_{j+1}"] = g_arr
        seed_runs.append(cell_samples)

    metrics = pool_across_seeds_k(seed_runs, g_true, K)
    # Add per-seed reduction_factor + vif so we can aggregate across mc_seeds
    # in the same shape as K=2 aggregator.
    per_seed_red = []
    per_seed_vif = []
    for r in seed_runs:
        m_seed = summarize_cell_k(r, g_true, K)
        per_seed_red.append(m_seed["reduction_factor_fast"])
        per_seed_vif.append(m_seed["vif_vs_fast"])
    return dict(
        K=K,
        reduction_factor=float(metrics["reduction_factor_fast"]),  # pooled
        reduction_min=float(np.min(per_seed_red)),
        reduction_max=float(np.max(per_seed_red)),
        vif_fast_mean=float(np.mean(per_seed_vif)),
        vif_fast_max=float(np.max(per_seed_vif)),
        per_seed_reduction=per_seed_red,
        per_seed_vif=per_seed_vif,
        bias_fast=float(metrics["bias_fast"]),
        bias_tL=float(metrics["bias_tL"]),
        var_fast=float(metrics["var_fast"]),
        var_tL=float(metrics["var_tL"]),
    )


# =============================================================================
# Per-topology K-curve aggregator
# =============================================================================


TOPOLOGY_FACTORIES: dict[str, Callable[[int], TabularMDP]] = {
    "canonical_3s2a": build_mdp_canonical_3s2a,
    "chain_5s2a": build_mdp_chain_5s2a,
    "cyclic_4s3a": build_mdp_cyclic_4s3a,
    "dense_5s3a": build_mdp_dense_5s3a,
    "terminal_3s2a": build_mdp_terminal_3s2a,
}

DEFAULT_TOPOLOGIES = list(TOPOLOGY_FACTORIES.keys())
DEFAULT_K_GRID = [2, 3, 5, 7, 10, 15, 20]
DEFAULT_MDP_SEEDS = [1337, 42, 1024, 7777, 31337]
DEFAULT_MC_SEEDS = [0, 1, 2]
PEAK_BAND_TOLERANCE = 0.05  # K within 5% of argmax mean reduction


def aggregate_topology_K_curve(
    topology_name: str, K_grid: list[int],
    mdp_seeds: list[int], mc_seeds: list[int],
    n_trials: int, T: int, theta_seed: int,
    delta_K2: int, cfg: RACConfig,
    noise_scale: float, decomposition_seed_base: int,
) -> dict[str, Any]:
    """Run all K values across all mdp_seeds for a single topology.

    Returns:
        per_K_per_mdp[K][mdp_seed_str] = cell aggregate (across mc_seeds)
        per_K_topology_aggregate[K] = mean across mdp_seeds
        peak_K_topology = K-value with max reduction_topology_mean
        peak_band_topology = K-values within PEAK_BAND_TOLERANCE of peak
    """
    factory = TOPOLOGY_FACTORIES[topology_name]

    per_K_per_mdp: dict[int, dict[str, dict[str, Any]]] = {K: {} for K in K_grid}
    per_K_topology_aggregate: dict[int, dict[str, float]] = {}

    rng_theta = np.random.default_rng(theta_seed)
    # Theta is sampled per-topology to match the topology's (n_states, n_actions)
    # shape; we use the same rng_theta seed for reproducibility but the shape
    # depends on the topology dims. (This matches V2's convention of
    # rng.normal(scale=0.3, size=(n_s, n_a)) — same scale, topology-specific shape.)
    sample_mdp = factory(seed=mdp_seeds[0])
    theta = rng_theta.normal(scale=0.3, size=(sample_mdp.n_states, sample_mdp.n_actions))

    for K in K_grid:
        for mdp_seed in mdp_seeds:
            base_mdp = factory(seed=mdp_seed)
            if K == 2:
                cell = aggregate_K2_per_mdp(
                    base_mdp=base_mdp, theta=theta, T=T,
                    n_trials=n_trials, delta_steps=delta_K2,
                    mc_seeds=mc_seeds, cfg=cfg,
                )
            else:
                deltas = [0] + list(range(1, K))
                cell = run_Kgeq3_cell(
                    base_mdp=base_mdp, theta=theta, K=K, T=T,
                    n_trials=n_trials, deltas=deltas,
                    mc_seeds=mc_seeds, cfg=cfg,
                    decomposition_seed=decomposition_seed_base + mdp_seed,
                    noise_scale=noise_scale,
                )
            per_K_per_mdp[K][f"mdp_{mdp_seed}"] = cell

        # Aggregate across mdp_seeds for this K
        red_per_mdp = [per_K_per_mdp[K][f"mdp_{m}"]["reduction_factor"]
                       for m in mdp_seeds]
        red_min_per_mdp = [per_K_per_mdp[K][f"mdp_{m}"]["reduction_min"]
                           for m in mdp_seeds]
        vif_per_mdp = [per_K_per_mdp[K][f"mdp_{m}"]["vif_fast_mean"]
                       for m in mdp_seeds]
        vif_max_per_mdp = [per_K_per_mdp[K][f"mdp_{m}"]["vif_fast_max"]
                           for m in mdp_seeds]
        per_K_topology_aggregate[K] = dict(
            K=K,
            reduction_topology_mean=float(np.mean(red_per_mdp)),
            reduction_topology_min=float(np.min(red_min_per_mdp)),
            reduction_topology_max=float(np.max(red_per_mdp)),
            vif_topology_mean=float(np.mean(vif_per_mdp)),
            vif_topology_max=float(np.max(vif_max_per_mdp)),
            per_mdp_reduction=red_per_mdp,
            per_mdp_reduction_min=red_min_per_mdp,
        )

    # Peak detection
    K_means = {K: per_K_topology_aggregate[K]["reduction_topology_mean"]
               for K in K_grid}
    peak_K = int(max(K_means, key=K_means.get))
    peak_value = K_means[peak_K]
    peak_band = sorted([
        K for K in K_grid
        if K_means[K] >= peak_value * (1 - PEAK_BAND_TOLERANCE)
    ])

    return dict(
        topology=topology_name,
        n_states=sample_mdp.n_states,
        n_actions=sample_mdp.n_actions,
        per_K_per_mdp=per_K_per_mdp,
        per_K_topology_aggregate=per_K_topology_aggregate,
        peak_K_topology=peak_K,
        peak_band_topology=peak_band,
        K_grid=K_grid,
        topology_red_min_overall=float(min(K_means.values())),
        topology_red_max_overall=float(max(K_means.values())),
    )


# =============================================================================
# Verdict logic — verbatim from PREREG sec 4 (FALSIFIED checked first)
# =============================================================================


def evaluate_branches(per_topology: dict[str, dict[str, Any]],
                       K_grid: list[int]) -> dict[str, Any]:
    """Apply PREREG sec 4 branch logic to the full 5-topology x 7-K matrix."""
    topologies = list(per_topology.keys())
    n_topologies = len(topologies)

    # Per-topology summaries
    peak_K_by_t = {t: per_topology[t]["peak_K_topology"] for t in topologies}
    peak_band_by_t = {t: per_topology[t]["peak_band_topology"] for t in topologies}
    red_min_by_t = {t: per_topology[t]["topology_red_min_overall"] for t in topologies}
    vif_max_by_t = {
        t: max(
            per_topology[t]["per_K_topology_aggregate"][K]["vif_topology_max"]
            for K in K_grid
        )
        for t in topologies
    }

    # Branch 4 — FALSIFIED-K-PEAK-NON-GENERALISED (checked FIRST)
    f4_peak_shift_left = any(peak_K_by_t[t] in {2, 3, 5} for t in topologies)
    f4_red_too_low = any(red_min_by_t[t] <= 1.5 for t in topologies)
    f4_peak_band_excludes_high = sum(
        not any(K in peak_band_by_t[t] for K in (10, 15, 20))
        for t in topologies
    ) >= 2
    falsified = f4_peak_shift_left or f4_red_too_low or f4_peak_band_excludes_high
    if falsified:
        verdict = "FALSIFIED-K-PEAK-NON-GENERALISED"
        verdict_reason = []
        if f4_peak_shift_left:
            verdict_reason.append(
                f"peak K shifted to small in topology(ies): "
                f"{[t for t in topologies if peak_K_by_t[t] in {2,3,5}]}"
            )
        if f4_red_too_low:
            verdict_reason.append(
                f"min reduction <=1.5 in topology(ies): "
                f"{[t for t in topologies if red_min_by_t[t] <= 1.5]}"
            )
        if f4_peak_band_excludes_high:
            verdict_reason.append(
                f">=2 topologies have peak_band excluding all of {{10,15,20}}: "
                f"{[t for t in topologies if not any(K in peak_band_by_t[t] for K in (10,15,20))]}"
            )
        return _build_verdict_dict(
            verdict, verdict_reason, peak_K_by_t, peak_band_by_t,
            red_min_by_t, vif_max_by_t, topologies,
        )

    # Branch 1 — ESTABLISHED-K-PEAK-CROSS-TOPOLOGY
    b1_all_high_peak = all(
        any(K in peak_band_by_t[t] for K in (10, 15, 20))
        for t in topologies
    )
    b1_all_red_ge_5 = all(red_min_by_t[t] >= 5.0 for t in topologies)
    b1_at_least_4_have_K15 = sum(
        15 in peak_band_by_t[t] for t in topologies
    ) >= 4
    b1_all_vif_lt_2 = all(vif_max_by_t[t] < 2.0 for t in topologies)
    if b1_all_high_peak and b1_all_red_ge_5 and b1_at_least_4_have_K15 and b1_all_vif_lt_2:
        verdict = "ESTABLISHED-K-PEAK-CROSS-TOPOLOGY"
        return _build_verdict_dict(
            verdict, ["all gates pass"],
            peak_K_by_t, peak_band_by_t, red_min_by_t, vif_max_by_t, topologies,
        )

    # Branch 2 — MODERATE-K-PEAK-ROBUST
    b2_at_least_3_have_K15 = sum(
        15 in peak_band_by_t[t] for t in topologies
    ) >= 3
    b2_all_red_ge_3 = all(red_min_by_t[t] >= 3.0 for t in topologies)
    b2_no_left_shift = not any(peak_K_by_t[t] in {2, 3, 5} for t in topologies)
    if b2_at_least_3_have_K15 and b2_all_red_ge_3 and b2_no_left_shift:
        verdict = "MODERATE-K-PEAK-ROBUST"
        return _build_verdict_dict(
            verdict,
            [f"K=15 in peak_band on {sum(15 in peak_band_by_t[t] for t in topologies)}/{n_topologies}"],
            peak_K_by_t, peak_band_by_t, red_min_by_t, vif_max_by_t, topologies,
        )

    # Branch 3 — PRELIMINARY-K-PEAK-TOPOLOGY-DEPENDENT
    b3_alt_topology_K15 = sum(
        15 in peak_band_by_t[t] for t in topologies if t != "canonical_3s2a"
    ) >= 1
    if b3_alt_topology_K15:
        verdict = "PRELIMINARY-K-PEAK-TOPOLOGY-DEPENDENT"
        return _build_verdict_dict(
            verdict,
            [f"At least 1 alternative topology has K=15 in peak band; "
             f"B1/B2 fail"],
            peak_K_by_t, peak_band_by_t, red_min_by_t, vif_max_by_t, topologies,
        )

    # Default fallback (no branch triggered) — should be rare; document explicitly
    verdict = "PRELIMINARY-K-PEAK-CANONICAL-ONLY"
    return _build_verdict_dict(
        verdict,
        ["No alternative topology has K=15 in peak band; B1/B2/B4 all fail"],
        peak_K_by_t, peak_band_by_t, red_min_by_t, vif_max_by_t, topologies,
    )


def _build_verdict_dict(verdict, reasons, peak_K_by_t, peak_band_by_t,
                         red_min_by_t, vif_max_by_t, topologies):
    return dict(
        verdict=verdict,
        verdict_reasons=reasons,
        peak_K_by_topology=peak_K_by_t,
        peak_band_by_topology={t: list(peak_band_by_t[t]) for t in topologies},
        topology_red_min={t: red_min_by_t[t] for t in topologies},
        topology_vif_max={t: vif_max_by_t[t] for t in topologies},
        n_topologies_with_K15_in_peak_band=sum(
            15 in peak_band_by_t[t] for t in topologies
        ),
        n_topologies_with_high_peak_band=sum(
            any(K in peak_band_by_t[t] for K in (10, 15, 20))
            for t in topologies
        ),
    )


# =============================================================================
# Figure (4 panels: per-topology K-curve, peak-K bar, vif heat, summary)
# =============================================================================


def make_figure(summary: dict[str, Any], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K_grid = summary["config"]["K_grid"]
    per_topology = summary["per_topology"]
    topologies = list(per_topology.keys())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    color_by_t = {t: colors[i % len(colors)] for i, t in enumerate(topologies)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=140)

    # Panel A — K-curve per topology
    ax = axes[0, 0]
    for t in topologies:
        red_means = [
            per_topology[t]["per_K_topology_aggregate"][K]["reduction_topology_mean"]
            for K in K_grid
        ]
        ax.plot(K_grid, red_means, "o-", color=color_by_t[t], label=t, linewidth=1.5)
        # Highlight peak
        peak_K = per_topology[t]["peak_K_topology"]
        peak_val = per_topology[t]["per_K_topology_aggregate"][peak_K]["reduction_topology_mean"]
        ax.scatter([peak_K], [peak_val], marker="*", s=200, color=color_by_t[t],
                   edgecolor="black", linewidth=0.8, zorder=5)
    ax.axhline(7.0, color="green", linestyle=":", lw=0.7, label="Branch 1 floor (7x)")
    ax.axhline(5.0, color="orange", linestyle=":", lw=0.7, label="Branch 1 cross-topology floor (5x)")
    ax.axhline(3.0, color="red", linestyle=":", lw=0.7, label="Branch 2 floor (3x)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(K_grid)
    ax.set_xticklabels([str(K) for K in K_grid])
    ax.set_xlabel("K (channel count)")
    ax.set_ylabel("reduction-factor mean (5 mdp-seeds x 3 mc-seeds = 15 reps)")
    ax.set_title("Panel A — K-curve by topology (* = topology peak)")
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.25, which="both")

    # Panel B — peak-K bar chart per topology
    ax = axes[0, 1]
    peak_Ks = [per_topology[t]["peak_K_topology"] for t in topologies]
    bars = ax.bar(topologies, peak_Ks,
                   color=[color_by_t[t] for t in topologies], alpha=0.8)
    for bar, peak_K, t in zip(bars, peak_Ks, topologies):
        peak_band = per_topology[t]["peak_band_topology"]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"K={peak_K}\nband={peak_band}",
                ha="center", va="bottom", fontsize=7)
    ax.axhline(15.0, color="black", linestyle="--", lw=0.7,
               label="canonical  peak (K=15)")
    ax.set_ylabel("argmax-K reduction-factor")
    ax.set_xticklabels(topologies, rotation=20, ha="right", fontsize=8)
    ax.set_title("Panel B — peak-K location per topology")
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.25, axis="y")

    # Panel C — VIF heatmap (topology x K)
    ax = axes[1, 0]
    vif_matrix = np.array([
        [per_topology[t]["per_K_topology_aggregate"][K]["vif_topology_max"]
         for K in K_grid]
        for t in topologies
    ])
    im = ax.imshow(vif_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=2.5)
    ax.set_xticks(range(len(K_grid)))
    ax.set_xticklabels([str(K) for K in K_grid])
    ax.set_yticks(range(len(topologies)))
    ax.set_yticklabels(topologies, fontsize=8)
    ax.set_xlabel("K")
    ax.set_title("Panel C — VIF-fast max (topology x K); Branch 1 ceiling = 2.0")
    for i, t in enumerate(topologies):
        for j, K in enumerate(K_grid):
            txt_color = "white" if vif_matrix[i, j] > 1.25 else "black"
            ax.text(j, i, f"{vif_matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color=txt_color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel D — verdict summary text
    ax = axes[1, 1]
    ax.axis("off")
    verdict = summary["verdict"]
    summary_text = (
        f"VERDICT: {verdict['verdict']}\n\n"
        f"Reasons:\n  - " + "\n  - ".join(verdict["verdict_reasons"]) + "\n\n"
        f"K=15 in peak_band: "
        f"{verdict['n_topologies_with_K15_in_peak_band']}/{len(topologies)} topologies\n"
        f"K in {{10,15,20}} in peak_band: "
        f"{verdict['n_topologies_with_high_peak_band']}/{len(topologies)} topologies\n\n"
        f"Per-topology peak K:\n"
    )
    for t in topologies:
        summary_text += (
            f"  {t}: peak={verdict['peak_K_by_topology'][t]} "
            f"band={verdict['peak_band_by_topology'][t]} "
            f"red_min={verdict['topology_red_min'][t]:.2f} "
            f"vif_max={verdict['topology_vif_max'][t]:.3f}\n"
        )
    summary_text += (
        f"\nWall: {summary['runtime_sec']:.1f}s\n"
        f"PREREG SHA: {summary['prereg_sha']}\n"
    )
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            verticalalignment="top", fontsize=8, family="monospace")

    fig.suptitle(
        " §4.5 cross-MDP-topology K-sweep replication "
        f"({len(topologies)} topologies x {len(K_grid)} K-values x "
        f"{summary['config']['n_trials']} trials)",
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--topologies", type=str, nargs="+", default=DEFAULT_TOPOLOGIES,
                   choices=DEFAULT_TOPOLOGIES,
                   help="Subset of topologies to sweep.")
    p.add_argument("--K-grid", type=int, nargs="+", default=DEFAULT_K_GRID,
                   help="K-values to sweep per topology.")
    p.add_argument("--mdp-seeds", type=int, nargs="+", default=DEFAULT_MDP_SEEDS,
                   help="MDP construction seeds per topology.")
    p.add_argument("--mc-seeds", type=int, nargs="+", default=DEFAULT_MC_SEEDS,
                   help="MC trajectory-pool seeds per (topology, K, mdp_seed).")
    p.add_argument("--n-trials", type=int, default=3000,
                   help="MC trajectories per cell (per mc_seed).")
    p.add_argument("--trajectory-len", type=int, default=50)
    p.add_argument("--theta-seed", type=int, default=7)
    p.add_argument("--delta-K2", type=int, default=20,
                   help="K=2 anchor delay (matches  /  47.9x).")
    p.add_argument("--tau-age", type=float, default=1000.0)
    p.add_argument("--is-clip", type=float, default=1.0)
    p.add_argument("--alpha-delta", type=float, default=1.0)
    p.add_argument("--noise-scale", type=float, default=0.3,
                   help="K-channel decomposition noise sigma.")
    p.add_argument("--decomposition-seed-base", type=int, default=20260426,
                   help="Base seed added to mdp_seed for noise decomposition.")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke run: K_grid=[2,3,5], 1 mdp-seed, 1 mc-seed, "
                        "n_trials=100, 2 topologies. ~30s. Verdict NOT binding.")
    p.add_argument("--results-dir", type=Path,
                   default=ROOT / "results" / "track2_K_sweep_cross_mdp_topology")
    p.add_argument("--figs-dir", type=Path, default=ROOT / "results" / "figs")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.smoke:
        print("[SMOKE MODE] Reduced grid for fast sanity check; verdict NOT binding.")
        args.topologies = ["canonical_3s2a", "chain_5s2a"]
        args.K_grid = [2, 3, 5]
        args.mdp_seeds = [1337]
        args.mc_seeds = [0]
        args.n_trials = 100

    t0 = time.time()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figs_dir.mkdir(parents=True, exist_ok=True)

    cfg = RACConfig(
        tau_age=args.tau_age,
        is_clip=args.is_clip,
        alpha_delta=args.alpha_delta,
        max_correction_norm=1e9,
    )

    print("=" * 96)
    print(" §4.5 cross-MDP-topology K-sweep replication")
    print("=" * 96)
    print(f"Topologies     = {args.topologies}")
    print(f"K-grid         = {args.K_grid}")
    print(f"MDP seeds      = {args.mdp_seeds}")
    print(f"MC seeds       = {args.mc_seeds}")
    print(f"trials/cell    = {args.n_trials}")
    print(f"theta_seed     = {args.theta_seed}")
    print(f"delta (K=2)    = {args.delta_K2}")
    print(f"tau_age        = {args.tau_age}; is_clip={args.is_clip}; "
          f"alpha_delta={args.alpha_delta}")
    print(f"noise_scale    = {args.noise_scale}")
    n_cells = len(args.topologies) * len(args.K_grid) * len(args.mdp_seeds)
    n_trajs_total = (len(args.topologies) * len(args.K_grid)
                      * len(args.mdp_seeds) * len(args.mc_seeds) * args.n_trials)
    print(f"Total cells    = {len(args.topologies)} x {len(args.K_grid)} x "
          f"{len(args.mdp_seeds)} = {n_cells}")
    print(f"Total trajs    = {n_trajs_total}")
    print("-" * 96)

    per_topology: dict[str, dict[str, Any]] = {}
    for topology_name in args.topologies:
        print(f"[{time.time()-t0:6.1f}s] Topology: {topology_name}", flush=True)
        t_topo = time.time()
        per_topology[topology_name] = aggregate_topology_K_curve(
            topology_name=topology_name,
            K_grid=args.K_grid,
            mdp_seeds=args.mdp_seeds,
            mc_seeds=args.mc_seeds,
            n_trials=args.n_trials,
            T=args.trajectory_len,
            theta_seed=args.theta_seed,
            delta_K2=args.delta_K2,
            cfg=cfg,
            noise_scale=args.noise_scale,
            decomposition_seed_base=args.decomposition_seed_base,
        )
        peak_K = per_topology[topology_name]["peak_K_topology"]
        peak_band = per_topology[topology_name]["peak_band_topology"]
        red_min = per_topology[topology_name]["topology_red_min_overall"]
        red_max = per_topology[topology_name]["topology_red_max_overall"]
        print(f"  done in {time.time()-t_topo:.1f}s; peak_K={peak_K} "
              f"peak_band={peak_band} red[min,max]=[{red_min:.2f}, {red_max:.2f}]",
              flush=True)

    # Per-topology K-curve table
    print("\n" + "=" * 96)
    print("Per-topology K-curve table (reduction-factor mean across 5 mdp-seeds x 3 mc-seeds)")
    print("=" * 96)
    header = f"{'topology':>16s}  | " + "  |  ".join(
        [f"K={K:>2d}" for K in args.K_grid]
    ) + "  |  peak"
    print(header)
    print("-" * len(header))
    for t in args.topologies:
        cells = []
        for K in args.K_grid:
            r = per_topology[t]["per_K_topology_aggregate"][K]["reduction_topology_mean"]
            cells.append(f"{r:>5.1f}x")
        peak_K = per_topology[t]["peak_K_topology"]
        cells.append(f"  K={peak_K}")
        print(f"{t:>16s}  | " + "  |  ".join(cells))

    # Verdict
    print("\n" + "=" * 96)
    print("PREREG branch verdict")
    print("=" * 96)
    verdict = evaluate_branches(per_topology, K_grid=args.K_grid)
    print(f"  VERDICT: {verdict['verdict']}")
    print(f"  Reasons: {verdict['verdict_reasons']}")
    print(f"  K=15 in peak_band: "
          f"{verdict['n_topologies_with_K15_in_peak_band']}/{len(args.topologies)}")
    print(f"  K in {{10,15,20}} in peak_band: "
          f"{verdict['n_topologies_with_high_peak_band']}/{len(args.topologies)}")
    for t in args.topologies:
        print(f"    {t}: peak_K={verdict['peak_K_by_topology'][t]} "
              f"band={verdict['peak_band_by_topology'][t]} "
              f"red_min={verdict['topology_red_min'][t]:.2f} "
              f"vif_max={verdict['topology_vif_max'][t]:.3f}")

    # Per-topology slice files
    for t in args.topologies:
        with open(args.results_dir / f"topology_{t}.json", "w") as f:
            json.dump(per_topology[t], f, indent=2, default=_json_default)

    # Master summary
    summary = dict(
        config=dict(
            topologies=args.topologies,
            K_grid=args.K_grid,
            mdp_seeds=args.mdp_seeds,
            mc_seeds=args.mc_seeds,
            n_trials=args.n_trials,
            trajectory_len=args.trajectory_len,
            theta_seed=args.theta_seed,
            delta_K2=args.delta_K2,
            tau_age=args.tau_age,
            is_clip=args.is_clip,
            alpha_delta=args.alpha_delta,
            noise_scale=args.noise_scale,
            decomposition_seed_base=args.decomposition_seed_base,
            smoke=args.smoke,
        ),
        prereg_sha="453363a",
        prereg_file="PREREG_T2_K_SWEEP_CROSS_MDP_TOPOLOGY.md",
        per_topology=per_topology,
        verdict=verdict,
        runtime_sec=time.time() - t0,
    )
    with open(args.results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nWrote {args.results_dir}/summary.json")
    print(f"Wrote {args.results_dir}/topology_*.json (n={len(args.topologies)})")

    fig_path = args.figs_dir / "track2_K_sweep_cross_mdp_topology.png"
    make_figure(summary, fig_path)
    print(f"Wrote {fig_path}")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
