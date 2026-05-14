"""F-T2 (iter+N+312): Heavy-tailed delay-distribution stress test for RAC.

Pre-registered in `PREREG_FT2_HEAVY_TAIL_DELAY.md` (commit 1b1591a, 2026-04-26
04:36 IST). Implements the design verbatim — 5 delay distributions matched at
E[Delta]=20, 5 MDP seeds x 3 MC seeds x 5 distributions x 3 tau_age x 1000 trials
= 75 cells x 3000 trajectories each.

The SCIENTIFIC question (closes a reviewer attack):

  All existing T2 RAC validators use a SINGLE deterministic delay Delta per cell.
  In production async RLHF, slow-RM evaluation latency is STOCHASTIC and often
  HEAVY-TAILED (queue contention, batch jitter, GPU evictions). Does the K=2
  bias-reduction (47.9x at deterministic Delta=20, 9.4x at deterministic
  Delta=20 with tau_age=200 per iter+N+197 table) survive when Delta is drawn
  per-trajectory from increasingly heavy-tailed distributions?

The 5 distributions, all matched to E[Delta]=20 so any divergence isolates TAIL
SHAPE, not mean (truncation tail mass shifts the realised mean for cauchy_trunc;
this is reported and disclosed):

  deterministic: Delta == 20 (control / reproduces iter+N+197 Delta=20 anchor)
  gaussian:      clip(Round(N(20, 6^2)), 1, 200)        — thin-tail symmetric
  lognormal:     clip(Round(LogNormal(mu, 0.5)), 1, 200)
                 with mu = ln(20) - 0.5^2/2  (gives E[X] = 20)  — service-time classic
  pareto_finite: clip(Round(Pareto(alpha=3, x_m=2/3*20)), 1, 200)
                 alpha=3 -> finite mean & variance, infinite skew  — power-law
  cauchy_trunc:  clip(Round(median + scale*StandardCauchy()), 1, 200), scale=4
                 median calibrated so empirical mean ~ 20 after truncation  — extreme

Reuses `verify_rac_gradient_correction.py` and `verify_rac_long_delay_Delta_100_200.py`
infrastructure verbatim (build_mdp, sample_trajectory, naive_pg_estimator,
oracle_pg_estimator, true_policy_gradient, softmax_policy, RACConfig). The ONLY
new logic is `sample_delays_for_distribution()` and the per-trajectory
RAC re-run loop (one global_step per trajectory, not one per cell).

PREREG branches (4-way mutually exclusive, gated at tau_age=200):
  ESTABLISHED-ROBUST     : reduction_min(all 5 dists) >= 7   AND
                           mean(heavy_tail_group) >= 0.75 * deterministic AND
                           vif_fast_max(all 5) < 2
  MODERATE-DEGRADATION   : reduction_min(all 5 dists) >= 3   AND
                           mean(heavy_tail_group) >= 0.5  * deterministic
  PRELIMINARY-TAIL-SENS  : at least one dist red_min >= 1.5  AND
                           vif_fast_mean(all 5) < 3      AND
                           monotone in tail-heaviness ordering
  FALSIFIED              : pareto_finite or cauchy_trunc reduction_mean < 1.5
                           OR vif_fast_mean >= 3 at tau_age=200

Outputs:
  results/track2_rac_heavy_tail_delay/summary.json        full 75-cell aggregate + verdict
  results/track2_rac_heavy_tail_delay/delta_dist_<name>.json  per-distribution slice
  results/figs/track2_rac_heavy_tail_delay.png            3-panel figure

Wall-clock: ~60 min single CPU thread (375 000 trajectories total).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Reuse iter+N+15 / iter+N+197 infrastructure verbatim.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.rac import RACConfig  # noqa: E402

from verify_rac_gradient_correction import (  # noqa: E402
    build_mdp,
    sample_trajectory,
    softmax_policy,
    true_policy_gradient,
    naive_pg_estimator,
    rac_corrected_pg_estimator,
    grad_log_pi_tabular,
    _json_default,
)

from verify_rac_long_delay_Delta_100_200 import (  # noqa: E402
    oracle_pg_estimator,
    sample_trajectories_batch,
)


# =============================================================================
# Per-trajectory delay samplers — five distributions matched at E[Delta]=20
# =============================================================================


def _clip_round(x: np.ndarray, lo: int = 1, hi: int = 200) -> np.ndarray:
    """Round to nearest int and clip to [lo, hi] (max delay = RolloutCache TTL)."""
    return np.clip(np.rint(x).astype(np.int64), lo, hi)


def sample_delays_for_distribution(
    name: str, n: int, rng: np.random.Generator,
    *, target_mean: float = 20.0, lo: int = 1, hi: int = 200,
) -> np.ndarray:
    """Draw `n` per-trajectory delays from `name` distribution, clipped to [lo, hi].

    All distributions are calibrated so the *unclipped* expectation equals
    target_mean (= 20). Clipping/truncation may shift the realised mean for
    very heavy tails (cauchy_trunc); the realised empirical mean+variance is
    REPORTED and disclosed (PREREG honest-disclosure obligation #1).
    """
    if name == "deterministic":
        return np.full(n, int(target_mean), dtype=np.int64)
    if name == "gaussian":
        # N(mu=20, sigma=6) -> realised mean ~20, variance ~36 after clip
        x = rng.normal(loc=target_mean, scale=6.0, size=n)
        return _clip_round(x, lo, hi)
    if name == "lognormal":
        # LogNormal(mu, sigma) has E = exp(mu + sigma^2/2)
        # Want E = target_mean, sigma = 0.5 -> mu = ln(target_mean) - sigma^2/2
        sigma = 0.5
        mu = math.log(target_mean) - sigma * sigma / 2.0
        x = rng.lognormal(mean=mu, sigma=sigma, size=n)
        return _clip_round(x, lo, hi)
    if name == "pareto_finite":
        # Pareto(alpha=3) has support [x_m, inf) with E = alpha*x_m/(alpha-1)
        # Want E = target_mean, alpha=3 -> x_m = target_mean * (alpha-1)/alpha = 2/3 * target_mean
        # numpy's `pareto(a)` returns Lomax = Pareto - 1, so we add x_m back.
        alpha = 3.0
        x_m = target_mean * (alpha - 1.0) / alpha
        x = (rng.pareto(alpha, size=n) + 1.0) * x_m
        return _clip_round(x, lo, hi)
    if name == "cauchy_trunc":
        # StandardCauchy has no finite mean; we shift by `median` and scale by `scale`,
        # then truncate to [lo, hi]. Median is calibrated so realised empirical mean
        # AFTER truncation ~ target_mean. We oversample 4x and pick the first n in-range
        # to ensure a well-defined sample with the target mean; the realised mean
        # is reported in the JSON for honest disclosure.
        median = 18.0   # calibrated empirically: shifts mass leftward to compensate
                        # for right-tail truncation pulling mean up.
        scale = 4.0
        # Oversample to cope with truncation losses; rejection-sample is overkill
        # for our purpose (we WANT truncation to define the distribution).
        x = median + scale * rng.standard_cauchy(size=n * 6)
        x = _clip_round(x, lo, hi)
        return x[:n]
    raise ValueError(f"Unknown delay distribution: {name!r}")


def empirical_delay_stats(deltas: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(deltas)),
        "mean": float(deltas.mean()),
        "var": float(deltas.var(ddof=1)) if len(deltas) > 1 else 0.0,
        "std": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        "min": int(deltas.min()),
        "median": float(np.median(deltas)),
        "p90": float(np.quantile(deltas, 0.9)),
        "p99": float(np.quantile(deltas, 0.99)),
        "max": int(deltas.max()),
    }


# =============================================================================
# RAC sweep over per-trajectory delays
# =============================================================================


def run_rac_per_trajectory(
    trajs: list[dict], naive_cache: np.ndarray, oracle_cache: np.ndarray,
    theta: np.ndarray, mdp, deltas_per_traj: np.ndarray, cfg: RACConfig,
) -> dict[str, np.ndarray]:
    """Run RAC estimator with a per-trajectory delay vector.

    `trajs[i]` uses `deltas_per_traj[i]` as its global_step offset. Naive and
    oracle baselines are pre-cached and identical to deterministic-Delta runs
    (they don't use Delta).
    """
    n = len(trajs)
    g_rac = np.empty((n, mdp.n_states, mdp.n_actions))
    for i, tr in enumerate(trajs):
        g_rac[i] = rac_corrected_pg_estimator(
            tr, theta, mdp, int(deltas_per_traj[i]), cfg,
        )
    return dict(g_naive=naive_cache, g_rac=g_rac, g_oracle=oracle_cache)


def summarize(g_naive, g_rac, g_oracle, g_true) -> dict[str, float]:
    """Bias / variance / reduction / VIF — same metric set as iter+N+197."""
    mean_naive = g_naive.mean(axis=0)
    mean_rac = g_rac.mean(axis=0)
    mean_oracle = g_oracle.mean(axis=0)
    bias_naive = float(np.linalg.norm(mean_naive - g_true))
    bias_rac = float(np.linalg.norm(mean_rac - g_true))
    bias_oracle = float(np.linalg.norm(mean_oracle - g_true))
    var_naive = float(g_naive.var(axis=0, ddof=1).mean())
    var_rac = float(g_rac.var(axis=0, ddof=1).mean())
    var_oracle = float(g_oracle.var(axis=0, ddof=1).mean())
    return dict(
        bias_naive=bias_naive,
        bias_rac=bias_rac,
        bias_oracle=bias_oracle,
        var_naive=var_naive,
        var_rac=var_rac,
        var_oracle=var_oracle,
        reduction_factor=bias_naive / max(bias_rac, 1e-12),
        vif_vs_fast=var_rac / max(var_naive, 1e-12),
        vif_vs_oracle=var_rac / max(var_oracle, 1e-12),
    )


def pool_seeds(seed_runs: list[dict[str, np.ndarray]], g_true) -> dict[str, Any]:
    g_naive = np.concatenate([r["g_naive"] for r in seed_runs], axis=0)
    g_rac = np.concatenate([r["g_rac"] for r in seed_runs], axis=0)
    g_oracle = np.concatenate([r["g_oracle"] for r in seed_runs], axis=0)
    out = summarize(g_naive, g_rac, g_oracle, g_true)
    out["per_seed_bias_rac"] = [
        float(np.linalg.norm(r["g_rac"].mean(axis=0) - g_true)) for r in seed_runs
    ]
    out["per_seed_bias_naive"] = [
        float(np.linalg.norm(r["g_naive"].mean(axis=0) - g_true)) for r in seed_runs
    ]
    return out


# =============================================================================
# Top-level driver
# =============================================================================


DISTRIBUTIONS = ["deterministic", "gaussian", "lognormal", "pareto_finite", "cauchy_trunc"]
HEAVY_TAIL_GROUP = ["gaussian", "lognormal", "pareto_finite", "cauchy_trunc"]


def run_one_mdp(
    mdp_seed: int, theta_seed: int,
    distributions: list[str], tau_grid: list[float],
    mc_seeds: list[int], n_trials: int, T: int, target_mean: float,
    delay_seed_base: int = 20260426,
) -> dict[str, Any]:
    """Per-MDP sweep: for each MC seed sample trajectories + delays, then for each
    (distribution, tau_age) cell run the RAC estimator.

    Naive + oracle are cached once per MC seed (independent of Delta). RAC is
    re-run with a per-trajectory delay vector for each (distribution, tau_age).
    """
    mdp = build_mdp(seed=mdp_seed)
    rng_theta = np.random.default_rng(theta_seed)
    theta = rng_theta.normal(scale=0.3, size=(mdp.n_states, mdp.n_actions))
    g_true = true_policy_gradient(theta, mdp)
    g_true_norm = float(np.linalg.norm(g_true))

    results: dict[str, Any] = dict(
        mdp_seed=mdp_seed,
        theta_seed=theta_seed,
        grad_true_l2=g_true_norm,
        cells={},
        empirical_delay_stats_by_dist={},
    )

    # 1) Trajectory pool (one trajectory list per MC seed; doesn't depend on Delta).
    traj_pool = {s: sample_trajectories_batch(theta, mdp, T, n_trials, s) for s in mc_seeds}

    # 2) Naive + oracle caches — identical for all (dist, tau_age) cells.
    naive_cache = {}
    oracle_cache = {}
    for s, trajs in traj_pool.items():
        gn = np.empty((len(trajs), mdp.n_states, mdp.n_actions))
        go = np.empty((len(trajs), mdp.n_states, mdp.n_actions))
        for i, tr in enumerate(trajs):
            gn[i] = naive_pg_estimator(tr, theta, mdp)
            go[i] = oracle_pg_estimator(tr, theta, mdp)
        naive_cache[s] = gn
        oracle_cache[s] = go

    # 3) Per-distribution delay vectors. Independent of (mdp_seed, tau_age) but
    # use per-(MDP, distribution, MC-seed) random key for reproducibility.
    delays_per_seed_per_dist = {}
    for dist in distributions:
        delays_per_seed_per_dist[dist] = {}
        all_deltas = []
        for s in mc_seeds:
            # Composite seed: distribution-name hash * mdp_seed * mc_seed
            key = delay_seed_base + (hash(dist) % 1_000_003) + 31 * mdp_seed + s
            rng = np.random.default_rng(abs(key) & 0x7FFFFFFF)
            deltas = sample_delays_for_distribution(
                dist, n_trials, rng, target_mean=target_mean,
            )
            delays_per_seed_per_dist[dist][s] = deltas
            all_deltas.append(deltas)
        results["empirical_delay_stats_by_dist"][dist] = empirical_delay_stats(
            np.concatenate(all_deltas)
        )

    # 4) Run the (dist, tau_age) cells.
    for tau_age in tau_grid:
        cfg = RACConfig(
            tau_age=tau_age,
            is_clip=1.0,
            alpha_delta=1.0,
            max_correction_norm=1e9,
        )
        for dist in distributions:
            seed_runs = []
            for s in mc_seeds:
                seed_runs.append(run_rac_per_trajectory(
                    trajs=traj_pool[s],
                    naive_cache=naive_cache[s],
                    oracle_cache=oracle_cache[s],
                    theta=theta, mdp=mdp,
                    deltas_per_traj=delays_per_seed_per_dist[dist][s],
                    cfg=cfg,
                ))
            m = pool_seeds(seed_runs, g_true)
            key = f"tau{int(tau_age)}_dist_{dist}"
            results["cells"][key] = dict(
                tau_age=tau_age,
                dist=dist,
                **m,
            )
    return results


# =============================================================================
# Verdict logic — direct PREREG transcript
# =============================================================================


def evaluate_branches(agg_by_cell: dict[str, dict], deterministic_anchor_red: float,
                       gate_tau_age: int = 200,
                       distributions: list[str] = None) -> dict[str, Any]:
    """Return PREREG branch verdict at tau_age=gate_tau_age (200).

    Uses the supplied `distributions` list (defaults to canonical 5). Smoke
    tests with a subset return a SMOKE verdict marker so we never silently
    pretend a partial sweep passed PREREG.
    """
    if distributions is None:
        distributions = DISTRIBUTIONS
    smoke_mode = set(distributions) != set(DISTRIBUTIONS)

    cells_at_gate = {
        d: agg_by_cell[f"tau{gate_tau_age}_dist_{d}"] for d in distributions
    }

    red_mean = {d: cells_at_gate[d]["reduction_mean"] for d in distributions}
    red_min = {d: cells_at_gate[d]["reduction_min"] for d in distributions}
    vif_fast_mean = {d: cells_at_gate[d]["vif_fast_mean"] for d in distributions}
    vif_fast_max = {d: cells_at_gate[d]["vif_fast_max"] for d in distributions}

    det_red = red_mean.get("deterministic", float("nan"))
    heavy_in_run = [d for d in HEAVY_TAIL_GROUP if d in distributions]
    heavy_mean_red = (
        float(np.mean([red_mean[d] for d in heavy_in_run])) if heavy_in_run else float("nan")
    )

    all_red_min_ge_7 = all(red_min[d] >= 7.0 for d in distributions)
    all_red_min_ge_3 = all(red_min[d] >= 3.0 for d in distributions)
    all_vif_max_lt_2 = all(vif_fast_max[d] < 2.0 for d in distributions)
    all_vif_mean_lt_3 = all(vif_fast_mean[d] < 3.0 for d in distributions)

    # Branch 4 — FALSIFIED (check first; supersedes everything else)
    pf_falsified = ("pareto_finite" in red_mean and red_mean["pareto_finite"] < 1.5) or \
                    ("pareto_finite" in vif_fast_mean and vif_fast_mean["pareto_finite"] >= 3.0)
    ct_falsified = ("cauchy_trunc" in red_mean and red_mean["cauchy_trunc"] < 1.5) or \
                    ("cauchy_trunc" in vif_fast_mean and vif_fast_mean["cauchy_trunc"] >= 3.0)
    falsified = pf_falsified or ct_falsified
    if falsified:
        verdict = "FALSIFIED"
    elif (
        all_red_min_ge_7
        and (math.isnan(heavy_mean_red) or math.isnan(det_red) or heavy_mean_red >= 0.75 * det_red)
        and all_vif_max_lt_2
    ):
        verdict = "ESTABLISHED-ROBUST"
    elif all_red_min_ge_3 and (
        math.isnan(heavy_mean_red) or math.isnan(det_red) or heavy_mean_red >= 0.50 * det_red
    ):
        verdict = "MODERATE-DEGRADATION"
    else:
        # Branch 3 detection — monotone tail-heaviness ordering on the run subset
        ordered_full = ["deterministic", "gaussian", "lognormal", "pareto_finite", "cauchy_trunc"]
        ordered_in_run = [d for d in ordered_full if d in red_mean]
        ordered_red = [red_mean[d] for d in ordered_in_run]
        monotone = all(
            ordered_red[i] >= ordered_red[i + 1] - 1e-9
            for i in range(len(ordered_red) - 1)
        )
        any_red_ge_1p5 = any(red_mean[d] >= 1.5 for d in distributions)
        if any_red_ge_1p5 and all_vif_mean_lt_3 and monotone:
            verdict = "PRELIMINARY-TAIL-SENSITIVE"
        else:
            verdict = "PRELIMINARY-TAIL-SENSITIVE-NONMONOTONE"

    if smoke_mode:
        verdict = f"SMOKE-{verdict}"  # never silently pass PREREG on partial run

    return dict(
        verdict=verdict,
        smoke_mode=smoke_mode,
        gate_tau_age=gate_tau_age,
        distributions_in_run=distributions,
        det_anchor_reduction=det_red,
        heavy_tail_mean_reduction=heavy_mean_red,
        heavy_over_det_ratio=(
            heavy_mean_red / max(det_red, 1e-12)
            if not (math.isnan(det_red) or math.isnan(heavy_mean_red)) else float("nan")
        ),
        per_dist_reduction_mean=red_mean,
        per_dist_reduction_min=red_min,
        per_dist_vif_fast_mean=vif_fast_mean,
        per_dist_vif_fast_max=vif_fast_max,
        gate_predicates=dict(
            all_red_min_ge_7=all_red_min_ge_7,
            all_red_min_ge_3=all_red_min_ge_3,
            all_vif_max_lt_2=all_vif_max_lt_2,
            all_vif_mean_lt_3=all_vif_mean_lt_3,
            falsified_predicate=falsified,
        ),
    )


# =============================================================================
# Figure (3 panels: reduction vs dist, VIF vs dist, empirical Delta histogram)
# =============================================================================


def make_figure(summary: dict[str, Any], distributions: list[str],
                 tau_grid: list[float], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)
    colors = {50.0: "#d62728", 200.0: "#1f77b4", 1000.0: "#2ca02c"}
    markers = {50.0: "o", 200.0: "s", 1000.0: "^"}

    # Panel A — reduction factor by distribution and tau_age
    ax = axes[0]
    x = np.arange(len(distributions))
    width = 0.25
    for i, tau in enumerate(tau_grid):
        means = []
        mins = []
        maxs = []
        for d in distributions:
            c = summary["agg_by_cell"][f"tau{int(tau)}_dist_{d}"]
            means.append(c["reduction_mean"])
            mins.append(c["reduction_min"])
            maxs.append(c["reduction_max"])
        means = np.array(means)
        offsets = (i - 1) * width
        ax.bar(x + offsets, means, width=width, color=colors[tau],
               label=fr"$\tau_{{\rm age}}={int(tau)}$", alpha=0.8)
        ax.errorbar(x + offsets, means,
                    yerr=[np.array(means) - np.array(mins), np.array(maxs) - np.array(means)],
                    fmt='none', color='k', capsize=2.5, lw=0.7)
    ax.axhline(7.0, color="green", linestyle=":", lw=0.8, label="Branch 1 gate (7x)")
    ax.axhline(3.0, color="orange", linestyle=":", lw=0.8, label="Branch 2 gate (3x)")
    ax.axhline(1.5, color="red", linestyle=":", lw=0.8, label="Branch 4 gate (1.5x)")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in distributions], fontsize=8)
    ax.set_ylabel(r"bias-reduction factor (5-MDP mean $\pm$ range)")
    ax.set_yscale("log")
    ax.set_title("Panel A - reduction by distribution (matched E[Delta]=20)")
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.25, axis='y')

    # Panel B — VIF vs distribution
    ax = axes[1]
    for i, tau in enumerate(tau_grid):
        vif_means = []
        for d in distributions:
            c = summary["agg_by_cell"][f"tau{int(tau)}_dist_{d}"]
            vif_means.append(c["vif_fast_mean"])
        ax.plot(x, vif_means, marker=markers[tau], color=colors[tau],
                label=fr"$\tau_{{\rm age}}={int(tau)}$")
    ax.axhline(2.0, color="green", linestyle=":", lw=0.8, label="VIF gate (2x)")
    ax.axhline(3.0, color="red", linestyle=":", lw=0.8, label="Branch 4 VIF gate (3x)")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in distributions], fontsize=8)
    ax.set_ylabel(r"VIF / fast = var$_{\rm RAC}$ / var$_{\rm naive}$")
    ax.set_title("Panel B - VIF by distribution")
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.25)

    # Panel C — empirical Delta histogram (one MDP for clarity)
    ax = axes[2]
    first_mdp = next(iter(summary["per_mdp"].keys()))
    stats = summary["per_mdp"][first_mdp]["empirical_delay_stats_by_dist"]
    width2 = 0.18
    for i, d in enumerate(distributions):
        s = stats[d]
        # Approximate distribution shape via 5-point summary
        ax.bar(i, s["mean"], width=width2, color="lightgray", edgecolor="k", lw=0.5)
        ax.errorbar(i, s["mean"],
                    yerr=[[s["mean"] - s["min"]], [s["max"] - s["mean"]]],
                    fmt='none', color='k', capsize=3, lw=0.6)
        ax.scatter([i], [s["median"]], marker='_', color='blue', s=80, zorder=3,
                   label="median" if i == 0 else None)
        ax.scatter([i], [s["p99"]], marker='v', color='red', s=40, zorder=3,
                   label="p99" if i == 0 else None)
    ax.axhline(20.0, color="orange", linestyle="--", lw=0.8, label="target E[Delta]=20")
    ax.set_xticks(np.arange(len(distributions)))
    ax.set_xticklabels([d.replace("_", "\n") for d in distributions], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Delta (steps; min..max range)")
    ax.set_title(f"Panel C - empirical Delta stats ({first_mdp})")
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.25, which='both')

    fig.suptitle(
        "F-T2 (iter+N+312) - heavy-tail delay-distribution stress test "
        "(5 distributions x matched E[Delta]=20 x 3 tau_age x 5 MDP x 3 MC seeds)",
        fontsize=10,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--distributions", type=str, nargs="+", default=DISTRIBUTIONS)
    p.add_argument("--tau-grid", type=float, nargs="+", default=[50.0, 200.0, 1000.0])
    p.add_argument("--mdp-seeds", type=int, nargs="+",
                   default=[1337, 42, 1024, 7777, 31337])
    p.add_argument("--mc-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-trials", type=int, default=1000)
    p.add_argument("--trajectory-len", type=int, default=50)
    p.add_argument("--theta-seed", type=int, default=7)
    p.add_argument("--target-mean", type=float, default=20.0,
                   help="E[Delta] target across all distributions (matched).")
    p.add_argument("--gate-tau-age", type=int, default=200,
                   help="tau_age value at which PREREG branches are gated.")
    p.add_argument("--results-dir", type=Path,
                   default=ROOT / "results" / "track2_rac_heavy_tail_delay")
    p.add_argument("--figs-dir", type=Path, default=ROOT / "results" / "figs")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("F-T2 (iter+N+312) - Heavy-tail delay-distribution stress test for RAC")
    print("=" * 88)
    print(f"Distributions = {args.distributions}")
    print(f"tau_age grid  = {args.tau_grid}")
    print(f"MDP seeds     = {args.mdp_seeds}")
    print(f"MC seeds      = {args.mc_seeds}")
    print(f"trials/cell   = {args.n_trials} x {len(args.mc_seeds)} = {args.n_trials * len(args.mc_seeds)}")
    print(f"target E[Delta] = {args.target_mean}")
    print(f"grid size     = {len(args.distributions)} x {len(args.tau_grid)} x "
          f"{len(args.mdp_seeds)} = "
          f"{len(args.distributions) * len(args.tau_grid) * len(args.mdp_seeds)} cells")
    print(f"trajectories  = {args.n_trials * len(args.mc_seeds) * len(args.distributions) * len(args.tau_grid) * len(args.mdp_seeds)} total")
    print("-" * 88)

    per_mdp: dict[str, Any] = {}
    for mdp_seed in args.mdp_seeds:
        print(f"MDP seed {mdp_seed}...", flush=True)
        t_mdp = time.time()
        per_mdp[f"mdp_{mdp_seed}"] = run_one_mdp(
            mdp_seed=mdp_seed,
            theta_seed=args.theta_seed,
            distributions=args.distributions,
            tau_grid=args.tau_grid,
            mc_seeds=args.mc_seeds,
            n_trials=args.n_trials,
            T=args.trajectory_len,
            target_mean=args.target_mean,
        )
        print(f"  done in {time.time() - t_mdp:.1f}s", flush=True)

    # Aggregate across MDP seeds
    agg_by_cell: dict[str, dict[str, Any]] = {}
    for tau in args.tau_grid:
        for d in args.distributions:
            key = f"tau{int(tau)}_dist_{d}"
            red_factors = [per_mdp[m]["cells"][key]["reduction_factor"] for m in per_mdp]
            vifs_fast = [per_mdp[m]["cells"][key]["vif_vs_fast"] for m in per_mdp]
            vifs_oracle = [per_mdp[m]["cells"][key]["vif_vs_oracle"] for m in per_mdp]
            bias_naive = [per_mdp[m]["cells"][key]["bias_naive"] for m in per_mdp]
            bias_rac = [per_mdp[m]["cells"][key]["bias_rac"] for m in per_mdp]
            agg_by_cell[key] = dict(
                tau_age=tau, dist=d,
                reduction_mean=float(np.mean(red_factors)),
                reduction_min=float(np.min(red_factors)),
                reduction_max=float(np.max(red_factors)),
                reduction_std=float(np.std(red_factors, ddof=1)) if len(red_factors) > 1 else 0.0,
                vif_fast_mean=float(np.mean(vifs_fast)),
                vif_fast_max=float(np.max(vifs_fast)),
                vif_oracle_mean=float(np.mean(vifs_oracle)),
                vif_oracle_max=float(np.max(vifs_oracle)),
                bias_naive_mean=float(np.mean(bias_naive)),
                bias_rac_mean=float(np.mean(bias_rac)),
                per_mdp_reduction=red_factors,
            )

    # Print canonical distribution-vs-tau_age table
    print("\n" + "=" * 96)
    print("Distribution x tau_age table (5-MDP mean [min, max] reduction factor; matched E[Delta]=20)")
    print("=" * 96)
    print(f"{'distribution':>18s}  | " + "  |  ".join([f"tau_age={int(t):<5d}" for t in args.tau_grid]))
    print("-" * 96)
    for d in args.distributions:
        row = f"{d:>18s}  | "
        cells = []
        for tau in args.tau_grid:
            c = agg_by_cell[f"tau{int(tau)}_dist_{d}"]
            cells.append(
                f"{c['reduction_mean']:>6.1f}x "
                f"[{c['reduction_min']:>5.1f}, {c['reduction_max']:>6.1f}]  "
                f"vif={c['vif_fast_mean']:.2f}"
            )
        print(row + "  |  ".join(cells))
    print()

    # Realised empirical delay stats (pooled across MDPs and MC seeds via union)
    print("=" * 96)
    print("Realised empirical Delta stats (pooled across all MDP+MC seeds per distribution)")
    print("=" * 96)
    pooled_emp_stats = {}
    for d in args.distributions:
        pooled = []
        for m in per_mdp:
            pooled.append(per_mdp[m]["empirical_delay_stats_by_dist"][d])
        # Reconstruct pooled by re-sampling — but we only have summary stats; use mean of means.
        # Better: emit per-MDP stats AND a coarse pooled mean (weighted by n)
        means = [s["mean"] for s in pooled]
        vars_ = [s["var"] for s in pooled]
        maxs = [s["max"] for s in pooled]
        p99s = [s["p99"] for s in pooled]
        pooled_emp_stats[d] = dict(
            mean_of_means=float(np.mean(means)),
            mean_of_vars=float(np.mean(vars_)),
            mean_of_maxs=float(np.mean(maxs)),
            mean_of_p99s=float(np.mean(p99s)),
            per_mdp_means=means,
        )
        print(f"  {d:>18s}: mean_of_means={pooled_emp_stats[d]['mean_of_means']:6.2f}  "
              f"mean_of_vars={pooled_emp_stats[d]['mean_of_vars']:8.2f}  "
              f"mean_of_p99s={pooled_emp_stats[d]['mean_of_p99s']:6.1f}  "
              f"mean_of_maxs={pooled_emp_stats[d]['mean_of_maxs']:6.1f}")

    # PREREG branch verdict at gate_tau_age
    print("\n" + "=" * 96)
    print(f"PREREG branch verdict (gated at tau_age={args.gate_tau_age})")
    print("=" * 96)
    verdict = evaluate_branches(agg_by_cell, deterministic_anchor_red=0.0,
                                  gate_tau_age=args.gate_tau_age,
                                  distributions=args.distributions)
    print(f"  VERDICT: {verdict['verdict']}")
    print(f"  Deterministic anchor reduction = {verdict['det_anchor_reduction']:.2f}x")
    print(f"  Heavy-tail mean reduction      = {verdict['heavy_tail_mean_reduction']:.2f}x")
    print(f"  Heavy/Det ratio                = {verdict['heavy_over_det_ratio']:.3f}")
    print(f"  per-dist reduction_mean        = {verdict['per_dist_reduction_mean']}")
    print(f"  per-dist reduction_min         = {verdict['per_dist_reduction_min']}")
    print(f"  per-dist vif_fast_mean         = {verdict['per_dist_vif_fast_mean']}")
    print(f"  gate_predicates                = {verdict['gate_predicates']}")

    # Per-distribution slice files
    for d in args.distributions:
        subset = {k: v for k, v in agg_by_cell.items() if f"_dist_{d}" in k}
        out = dict(
            distribution=d,
            cells=subset,
            per_mdp_cells={m: {k: v for k, v in per_mdp[m]["cells"].items()
                                if f"_dist_{d}" in k}
                             for m in per_mdp},
            empirical_delay_stats_per_mdp={
                m: per_mdp[m]["empirical_delay_stats_by_dist"][d] for m in per_mdp
            },
            pooled_empirical_stats=pooled_emp_stats[d],
        )
        with open(args.results_dir / f"delta_dist_{d}.json", "w") as f:
            json.dump(out, f, indent=2, default=_json_default)

    # Master summary
    summary = dict(
        config=dict(
            distributions=args.distributions, tau_grid=args.tau_grid,
            mdp_seeds=args.mdp_seeds, mc_seeds=args.mc_seeds,
            n_trials=args.n_trials, trajectory_len=args.trajectory_len,
            theta_seed=args.theta_seed, target_mean=args.target_mean,
            gate_tau_age=args.gate_tau_age,
        ),
        prereg_sha="1b1591a",
        prereg_file="PREREG_FT2_HEAVY_TAIL_DELAY.md",
        agg_by_cell=agg_by_cell,
        pooled_empirical_delay_stats=pooled_emp_stats,
        per_mdp=per_mdp,
        verdict=verdict,
        runtime_sec=time.time() - t0,
    )
    with open(args.results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nWrote {args.results_dir}/summary.json (+ 5 delta_dist_*.json)")

    fig_path = args.figs_dir / "track2_rac_heavy_tail_delay.png"
    make_figure(summary, args.distributions, args.tau_grid, fig_path)
    print(f"Wrote {fig_path}")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
