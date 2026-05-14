"""Long-delay stress test for the RAC δ-injection primitive.

Extends ``verify_rac_gradient_correction.py`` (Δ∈{5,20,50}, τ_age=1000 → 47.9×
mean bias reduction headline) to Δ∈{100, 200} — 2-10× longer than the paper's
current sweep — to answer the ICLR-style reviewer question: "how far does
RAC's bias-reduction factor scale with delay, and does the age-discount
schedule w_age(Δ)=exp(−Δ/τ_age) dominate scaling behaviour at long delays?"

Setup
-----
- 5 MDP seeds: {1337, 42, 1024, 7777, 31337} — matches the K=2 MDP ablation
  in ``results/k2_mdp_seed_ablation/`` so the per-MDP range is
  comparable to the paper's existing headline.
- 3 MC seeds per MDP × 1000 trials = 3000 trajectories per cell — matches
  the paper's headline budget (verify_rac_gradient_correction.py default).
- Full Δ sweep: {5, 20, 50, 100, 200} so we can build the scaling table
  reviewers expect (existing 3 Δ + 2 new long-delay points).
- τ_age grid: {50 (production default), 200 (matched to Δ_max=200), 1000
  (reference — reproduces the paper's 47.9× headline up to MC noise at
  short Δ and exposes the pure correction-magnitude math at long Δ).
- For each cell we also compute the "oracle independent-sum" estimator
  that uses G_fast + G_slow directly as advantages — this is the
  bias-free upper bound on what RAC could achieve, and its variance is
  the natural "VIF vs independent-sum" denominator.

Hypothesis (from task spec, confirmed by w_age math)
----------------------------------------------------
- At τ_age=50, w_age(Δ=100)=e^{-2}≈0.135, w_age(Δ=200)=e^{-4}≈0.018 →
  RAC recovers only ~13.5%/1.8% of the slow-channel correction, so the
  reduction factor should DECAY roughly as (1 − w_age(Δ))^{-1} relative
  to the Δ=5 anchor. Expect ≤5× at Δ=200.
- At τ_age=200, w_age(100)=0.607, w_age(200)=0.368 → correction still
  substantial, reduction factor should remain ≥10× at Δ=200.
- At τ_age=1000, w_age is effectively 1 up to Δ=200, so reduction factor
  should be statistically indistinguishable from the Δ∈{5,20,50} paper
  headline (~30-80×, pooled mean 47.9×).

Honest scope
------------
If the data confirm the hypothesis, the paper gets a clean scope claim:
"RAC maintains its K=2 bias-reduction factor ≥X× whenever τ_age ≥ 2·Δ_max."
If RAC fails even at τ_age=200 (for example because of residual MC noise
or the base-rate numerator collapsing to the noise floor), the paper
reports that honestly as a scope boundary.

Budget
------
5 Δ × 3 τ_age × 5 MDP × 3 seeds × 1000 trials = 225 000 trajectories.
Each trajectory is 50 steps tabular, <1 ms — wall-clock <30 min on
a single CPU core, matching the task spec.

Usage
-----
    python scripts/verify_rac_long_delay_Delta_100_200.py                 # full grid
    python scripts/verify_rac_long_delay_Delta_100_200.py --n-trials 200  # smoke

Outputs
-------
    results/rac_long_delay/delta_100.json    # τ_age ablation @ Δ=100
    results/rac_long_delay/delta_200.json    # τ_age ablation @ Δ=200
    results/rac_long_delay/summary.json      # full sweep table + verdict
    results/figs/rac_long_delay_sweep.png    # Δ-sweep + τ_age panel
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Reuse the existing validator's MDP + estimators rather than re-implementing.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.rac import RACConfig  # noqa: E402
from verify_rac_gradient_correction import (  # noqa: E402
    build_mdp,
    true_policy_gradient,
    sample_trajectory,
    naive_pg_estimator,
    rac_corrected_pg_estimator,
    grad_log_pi_tabular,
    softmax_policy,
    _json_default,
)


# =============================================================================
# Oracle "independent-sum" estimator — bias-free upper bound on RAC
# =============================================================================


def oracle_pg_estimator(trajectory: dict[str, np.ndarray], theta: np.ndarray,
                         mdp) -> np.ndarray:
    """Oracle MC PG estimator using G_fast + G_slow = G_total directly.

    ĝ_oracle = (1/T) Σ_t G_total_t · ∇log π(a_t|s_t)

    This is an unbiased MC estimator of ∇J(θ) — its variance is the natural
    denominator for the "VIF vs independent-sum" metric: if RAC's variance
    is within 2× of oracle, we know the IS-clip + age-discount machinery is
    not inflating variance beyond what a causally-impossible oracle would
    already have to pay.
    """
    pi = softmax_policy(theta)
    T = len(trajectory["states"])
    g = np.zeros_like(theta)
    G_total = trajectory["G_fast"] + trajectory["G_slow"]
    for t in range(T):
        s, a = int(trajectory["states"][t]), int(trajectory["actions"][t])
        g += G_total[t] * grad_log_pi_tabular(
            s, a, pi, mdp.n_states, mdp.n_actions,
        )
    return g / T


# =============================================================================
# Monte-Carlo harness — with oracle baseline
# =============================================================================


def sample_trajectories_batch(theta, mdp, T, n_trials, seed):
    """Sample and cache n_trials trajectories once; reused across (Δ, τ_age)."""
    rng = np.random.default_rng(seed * 10_007 + 13)
    trajs = [sample_trajectory(theta, mdp, T, rng) for _ in range(n_trials)]
    return trajs


def run_cell_from_trajs(trajs, theta, mdp, delta_steps: int,
                         cfg: RACConfig) -> dict[str, np.ndarray]:
    """Re-use cached trajectories for (Δ, τ_age) — only RAC branch is re-run."""
    n_trials = len(trajs)
    g_naive = np.empty((n_trials, mdp.n_states, mdp.n_actions))
    g_rac = np.empty((n_trials, mdp.n_states, mdp.n_actions))
    g_oracle = np.empty((n_trials, mdp.n_states, mdp.n_actions))
    for i, traj in enumerate(trajs):
        g_naive[i] = naive_pg_estimator(traj, theta, mdp)
        g_rac[i] = rac_corrected_pg_estimator(traj, theta, mdp, delta_steps, cfg)
        g_oracle[i] = oracle_pg_estimator(traj, theta, mdp)
    return dict(g_naive=g_naive, g_rac=g_rac, g_oracle=g_oracle)


def run_cell_from_traj_and_naive(trajs, g_naive_cached, g_oracle_cached,
                                   theta, mdp, delta_steps, cfg):
    """Fastest path — naive and oracle already computed; only RAC branch runs."""
    n_trials = len(trajs)
    g_rac = np.empty((n_trials, mdp.n_states, mdp.n_actions))
    for i, traj in enumerate(trajs):
        g_rac[i] = rac_corrected_pg_estimator(traj, theta, mdp, delta_steps, cfg)
    return dict(g_naive=g_naive_cached, g_rac=g_rac, g_oracle=g_oracle_cached)


def summarize(g_naive, g_rac, g_oracle, g_true) -> dict[str, float]:
    """Compute bias / variance / reduction / VIF-fast / VIF-oracle."""
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


def pool_seeds(seed_runs: list[dict[str, np.ndarray]], g_true):
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
# Top-level sweep driver
# =============================================================================


def run_one_mdp(mdp_seed: int, theta_seed: int, deltas: list[int],
                 tau_grid: list[float], mc_seeds: list[int], n_trials: int,
                 T: int) -> dict[str, Any]:
    """Full Δ × τ_age sweep for one MDP seed."""
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
    )

    # 1) Sample trajectories once per MC seed (independent of Δ and τ_age).
    traj_pool: dict[int, list] = {
        s: sample_trajectories_batch(theta, mdp, T, n_trials, s)
        for s in mc_seeds
    }

    # 2) Compute naive + oracle once per MC seed (independent of Δ and τ_age).
    naive_cache: dict[int, np.ndarray] = {}
    oracle_cache: dict[int, np.ndarray] = {}
    for s, trajs in traj_pool.items():
        gn = np.empty((len(trajs), mdp.n_states, mdp.n_actions))
        go = np.empty((len(trajs), mdp.n_states, mdp.n_actions))
        for i, tr in enumerate(trajs):
            gn[i] = naive_pg_estimator(tr, theta, mdp)
            go[i] = oracle_pg_estimator(tr, theta, mdp)
        naive_cache[s] = gn
        oracle_cache[s] = go

    # 3) RAC estimator re-run per (τ_age, Δ) on the cached trajectories.
    for tau_age in tau_grid:
        cfg = RACConfig(
            tau_age=tau_age,
            is_clip=1.0,           # clamp ρ to [0,2]; identity actor → ρ=1
            alpha_delta=1.0,
            max_correction_norm=1e9,  # disable clamp — raw math
        )
        for delta_steps in deltas:
            seed_runs = [
                run_cell_from_traj_and_naive(
                    traj_pool[s], naive_cache[s], oracle_cache[s],
                    theta, mdp, delta_steps, cfg,
                )
                for s in mc_seeds
            ]
            m = pool_seeds(seed_runs, g_true)
            key = f"tau{int(tau_age)}_delta{delta_steps}"
            results["cells"][key] = dict(
                tau_age=tau_age,
                delta=delta_steps,
                w_age=float(np.exp(-delta_steps / tau_age)),
                **m,
            )
    return results


# =============================================================================
# Figure
# =============================================================================


def make_figure(summary: dict[str, Any], deltas: list[int],
                 tau_grid: list[float], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    colors = {50.0: "#d62728", 200.0: "#1f77b4", 1000.0: "#2ca02c"}
    markers = {50.0: "o", 200.0: "s", 1000.0: "^"}

    ax = axes[0]
    for tau in tau_grid:
        red_means = []
        red_mins = []
        red_maxs = []
        for d in deltas:
            vals = []
            for mdp_key, mdp_res in summary["per_mdp"].items():
                k = f"tau{int(tau)}_delta{d}"
                vals.append(mdp_res["cells"][k]["reduction_factor"])
            red_means.append(np.mean(vals))
            red_mins.append(np.min(vals))
            red_maxs.append(np.max(vals))
        red_means = np.array(red_means)
        red_mins = np.array(red_mins)
        red_maxs = np.array(red_maxs)
        ax.plot(deltas, red_means, marker=markers[tau], color=colors[tau],
                label=rf"$\tau_{{\rm age}}={int(tau)}$")
        ax.fill_between(deltas, red_mins, red_maxs, color=colors[tau], alpha=0.15)
    ax.axhline(3.0, color="k", linestyle="--", lw=0.8, label="gate (3×)")
    ax.set_xlabel(r"delay $\Delta$ (optimizer steps)")
    ax.set_ylabel(r"bias reduction factor $\|\mathrm{bias}_{\rm naive}\|/\|\mathrm{bias}_{\rm RAC}\|$")
    ax.set_yscale("log")
    ax.set_title("Panel A — RAC reduction factor vs Δ (5 MDP seeds)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for tau in tau_grid:
        vif_f = []
        vif_o = []
        for d in deltas:
            fv = []
            ov = []
            for mdp_key, mdp_res in summary["per_mdp"].items():
                k = f"tau{int(tau)}_delta{d}"
                fv.append(mdp_res["cells"][k]["vif_vs_fast"])
                ov.append(mdp_res["cells"][k]["vif_vs_oracle"])
            vif_f.append(np.mean(fv))
            vif_o.append(np.mean(ov))
        ax.plot(deltas, vif_f, marker=markers[tau], color=colors[tau],
                label=rf"VIF/fast $\tau={int(tau)}$", linestyle="-")
        ax.plot(deltas, vif_o, marker=markers[tau], color=colors[tau],
                label=rf"VIF/oracle $\tau={int(tau)}$", linestyle=":",
                alpha=0.7)
    ax.axhline(2.0, color="k", linestyle="--", lw=0.8, label="gate (2×)")
    ax.set_xlabel(r"delay $\Delta$ (optimizer steps)")
    ax.set_ylabel(r"VIF $= \mathrm{var}_{\rm RAC}/\mathrm{var}_{\rm baseline}$")
    ax.set_title("Panel B — VIF vs fast & vs oracle (5-MDP mean)")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Track 2 — RAC long-delay stress test (Δ ∈ {5,20,50,100,200}; τ_age ∈ {50,200,1000})",
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--deltas", type=int, nargs="+", default=[5, 20, 50, 100, 200])
    p.add_argument("--tau-grid", type=float, nargs="+", default=[50.0, 200.0, 1000.0])
    p.add_argument("--mdp-seeds", type=int, nargs="+",
                   default=[1337, 42, 1024, 7777, 31337])
    p.add_argument("--mc-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-trials", type=int, default=1000,
                   help="per (Δ, τ_age, MDP, MC seed). 1000 × 3 seeds = 3000/cell.")
    p.add_argument("--trajectory-len", type=int, default=50)
    p.add_argument("--theta-seed", type=int, default=7)
    p.add_argument("--results-dir", type=Path,
                   default=ROOT / "results" / "rac_long_delay")
    p.add_argument("--figs-dir", type=Path, default=ROOT / "results" / "figs")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Track 2 — RAC long-delay stress test (Δ=100, 200; 5 MDP seeds; 3 τ_age configs)")
    print("=" * 88)
    print(f"Δ grid       = {args.deltas}")
    print(f"τ_age grid   = {args.tau_grid}")
    print(f"MDP seeds    = {args.mdp_seeds}")
    print(f"MC seeds     = {args.mc_seeds}")
    print(f"trials/cell  = {args.n_trials} × {len(args.mc_seeds)} = "
          f"{args.n_trials * len(args.mc_seeds)} (paper headline budget)")
    print(f"grid size    = {len(args.deltas)} × {len(args.tau_grid)} × "
          f"{len(args.mdp_seeds)} = "
          f"{len(args.deltas) * len(args.tau_grid) * len(args.mdp_seeds)} cells")
    print("-" * 88)

    per_mdp: dict[str, Any] = {}
    for mdp_seed in args.mdp_seeds:
        print(f"MDP seed {mdp_seed}...", flush=True)
        t_mdp = time.time()
        per_mdp[f"mdp_{mdp_seed}"] = run_one_mdp(
            mdp_seed=mdp_seed,
            theta_seed=args.theta_seed,
            deltas=args.deltas,
            tau_grid=args.tau_grid,
            mc_seeds=args.mc_seeds,
            n_trials=args.n_trials,
            T=args.trajectory_len,
        )
        print(f"  done in {time.time() - t_mdp:.1f}s", flush=True)

    # Aggregate across MDP seeds
    agg_by_cell: dict[str, dict[str, Any]] = {}
    for tau in args.tau_grid:
        for d in args.deltas:
            key = f"tau{int(tau)}_delta{d}"
            red_factors = [per_mdp[m]["cells"][key]["reduction_factor"]
                           for m in per_mdp]
            vifs_fast = [per_mdp[m]["cells"][key]["vif_vs_fast"] for m in per_mdp]
            vifs_oracle = [per_mdp[m]["cells"][key]["vif_vs_oracle"] for m in per_mdp]
            bias_naive = [per_mdp[m]["cells"][key]["bias_naive"] for m in per_mdp]
            bias_rac = [per_mdp[m]["cells"][key]["bias_rac"] for m in per_mdp]
            w_age = per_mdp[f"mdp_{args.mdp_seeds[0]}"]["cells"][key]["w_age"]
            agg_by_cell[key] = dict(
                tau_age=tau, delta=d, w_age=w_age,
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

    # Print the canonical Δ × τ_age table
    print("\n" + "=" * 88)
    print("Δ-sweep × τ_age ablation table (5-MDP mean [min, max] reduction factor)")
    print("=" * 88)
    header = f"{'Δ':>6s} | " + " | ".join(
        f"τ={int(t):<4d} (w_age={np.exp(-args.deltas[0]/t):.2f}→{np.exp(-args.deltas[-1]/t):.2f})"
        for t in args.tau_grid
    )
    # Simpler header
    print(f"{'Δ':>4s}  | " + "  |  ".join([f"τ_age={int(t):<5d}" for t in args.tau_grid]))
    print("-" * 88)
    for d in args.deltas:
        row = f"{d:>4d}  | "
        cells = []
        for tau in args.tau_grid:
            c = agg_by_cell[f"tau{int(tau)}_delta{d}"]
            cells.append(
                f"{c['reduction_mean']:>6.1f}× "
                f"[{c['reduction_min']:>5.1f}, {c['reduction_max']:>6.1f}]  "
                f"(w={c['w_age']:.3f})"
            )
        print(row + "  |  ".join(cells))
    print()

    # Verdicts per τ_age at long delays
    print("=" * 88)
    print("Long-delay verdicts (Δ ∈ {100, 200})")
    print("=" * 88)
    for tau in args.tau_grid:
        for d in [100, 200]:
            if d not in args.deltas:
                continue
            c = agg_by_cell[f"tau{int(tau)}_delta{d}"]
            gate_bias = c["reduction_min"] >= 3.0
            gate_vif = c["vif_fast_max"] < 2.0
            ok = "PASS" if (gate_bias and gate_vif) else "FAIL"
            print(f"  τ_age={int(tau):<5d} Δ={d:<3d}  red={c['reduction_mean']:>6.1f}× "
                  f"[{c['reduction_min']:>5.1f}, {c['reduction_max']:>6.1f}]  "
                  f"VIF/fast={c['vif_fast_mean']:.2f}  VIF/oracle={c['vif_oracle_mean']:.2f}  "
                  f"{ok} (gate red≥3, VIF<2)")

    # Persist delta_100 / delta_200 / summary
    for d_target in [100, 200]:
        if d_target not in args.deltas:
            continue
        subset = {k: v for k, v in agg_by_cell.items() if f"_delta{d_target}" in k}
        out = dict(
            delta=d_target,
            cells=subset,
            per_mdp_cells={m: {k: v for k, v in per_mdp[m]["cells"].items()
                                if f"_delta{d_target}" in k}
                             for m in per_mdp},
            config=dict(
                mdp_seeds=args.mdp_seeds,
                mc_seeds=args.mc_seeds,
                n_trials=args.n_trials,
                trajectory_len=args.trajectory_len,
                tau_grid=args.tau_grid,
            ),
        )
        with open(args.results_dir / f"delta_{d_target}.json", "w") as f:
            json.dump(out, f, indent=2, default=_json_default)

    summary = dict(
        config=dict(
            deltas=args.deltas, tau_grid=args.tau_grid,
            mdp_seeds=args.mdp_seeds, mc_seeds=args.mc_seeds,
            n_trials=args.n_trials, trajectory_len=args.trajectory_len,
            theta_seed=args.theta_seed,
        ),
        agg_by_cell=agg_by_cell,
        per_mdp=per_mdp,
        runtime_sec=time.time() - t0,
    )
    with open(args.results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nWrote {args.results_dir}/delta_100.json, delta_200.json, summary.json")

    fig_path = args.figs_dir / "rac_long_delay_sweep.png"
    make_figure(summary, args.deltas, args.tau_grid, fig_path)
    print(f"Wrote {fig_path}")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
