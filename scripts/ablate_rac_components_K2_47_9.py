"""B2 component ablation for the Tab 1 K=2 47.9x headline (RLxF P2).

Reproduces the exact 47.9x configuration and ablates the three knobs in
the RAC correction formula

    A_i^total = A_i^partial + w_age(Delta) * rho_i^clip * alpha_delta * (r_slow_i - m_t)

so the closed-form K=2 row of Tab 1 in
`paper/workshop_T2_RLxF/rlxf_T2_iter1_ICML2026.tex`
gains a small ablation block isolating which knob carries the bias-
reduction. Audit: iter+N+367_RLxF_Nanda_audit_1209_IST.md sec B2.

The 47.9x reproduction uses:

    build_mdp(seed=1337), theta_seed=7,
    tau_age=1000, is_clip=1.0, alpha_delta=1.0, IdentityActor (rho=1),
    Delta in {5, 20, 50}, n_trials=1000, seeds={0,1,2}, T=50.

Components ablated (each cell holds the other two at the headline values):

    (a) FULL          tau_age=1000, is_clip=1.0,  alpha=1   (= 47.9x)
    (b) -w_age        tau_age=1e12, is_clip=1.0,  alpha=1   (no age decay)
    (c) bare additive tau_age=1e12, is_clip=1e12, alpha=1   (no clip, no decay)

At Delta=50 with tau_age=1000, w_age = exp(-50/1000) = 0.951. So removing
w_age slightly increases the magnitude of the correction. Removing both
w_age and the clip (the rho->1 IdentityActor already pins rho_clip=1, so
this knob is degenerate at the closed-form benchmark) ought to converge
on the bare additive correction. The headline 47.9x IS effectively the
bare additive correction with a 5% w_age damping.

Bootstrap 95% CI per cell uses the per-trajectory naive/RAC bias arrays
resampled with B=1000 replicates, matching `verify_rac_gradient_correction`
practice.

Outputs:
    results/track2_b2_ablation_47_9/ablation.json

Usage:
    python scripts/ablate_rac_components_K2_47_9.py

Runtime: ~3-5 min on a single CPU core (3 cells x 3 Delta x 3 seeds x
1000 trajectories = 27000 trajectories total).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import verify_rac_gradient_correction as V2  # noqa: E402
from src.rac import RACConfig  # noqa: E402


def bootstrap_ci_reduction(
    g_naive: np.ndarray,
    g_rac: np.ndarray,
    g_true: np.ndarray,
    n_boot: int = 1000,
    rng_seed: int = 17,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for the bias-reduction factor.

    Resamples trajectories WITH replacement, recomputes
    bias_naive/bias_rac on each replicate, and returns
    (point, lo, hi) at 2.5% / 97.5%.
    """
    rng = np.random.default_rng(rng_seed)
    n = g_naive.shape[0]
    point_naive = float(np.linalg.norm(g_naive.mean(axis=0) - g_true))
    point_rac = float(np.linalg.norm(g_rac.mean(axis=0) - g_true))
    point = point_naive / max(point_rac, 1e-12)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bn = float(np.linalg.norm(g_naive[idx].mean(axis=0) - g_true))
        br = float(np.linalg.norm(g_rac[idx].mean(axis=0) - g_true))
        reps[b] = bn / max(br, 1e-12)
    lo = float(np.percentile(reps, 2.5))
    hi = float(np.percentile(reps, 97.5))
    return point, lo, hi


def run_cell(
    theta: np.ndarray,
    mdp: V2.TabularMDP,
    cfg: RACConfig,
    deltas: list[int],
    seeds: list[int],
    n_trials: int,
    T: int,
    g_true: np.ndarray,
) -> dict:
    """Run one (cfg) cell across Delta x seeds, pool, return summary."""
    per_delta: dict[int, dict] = {}
    g_naive_pool = []
    g_rac_pool = []
    for delta_steps in deltas:
        seed_runs = []
        for seed in seeds:
            seed_runs.append(V2.run_mc_cell(
                theta=theta, mdp=mdp, T=T, n_trials=n_trials,
                delta_steps=delta_steps, seed=seed, cfg=cfg,
            ))
        g_naive = np.concatenate([r["g_naive"] for r in seed_runs], axis=0)
        g_rac = np.concatenate([r["g_rac"] for r in seed_runs], axis=0)
        m = V2.summarize_cell(g_naive, g_rac, g_true)
        per_delta[delta_steps] = m
        g_naive_pool.append(g_naive)
        g_rac_pool.append(g_rac)
    g_naive_all = np.concatenate(g_naive_pool, axis=0)
    g_rac_all = np.concatenate(g_rac_pool, axis=0)
    point, lo, hi = bootstrap_ci_reduction(g_naive_all, g_rac_all, g_true)
    red_factors = [m["reduction_factor"] for m in per_delta.values()]
    return {
        "per_delta": {str(k): v for k, v in per_delta.items()},
        "mean_reduction": float(np.mean(red_factors)),
        "min_reduction": float(np.min(red_factors)),
        "max_reduction": float(np.max(red_factors)),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "ci_point": point,
    }


def main() -> int:
    t0 = time.time()

    # Reproduce the 47.9x setup verbatim.
    DELTAS = [5, 20, 50]
    SEEDS = [0, 1, 2]
    N_TRIALS = 1000
    T = 50

    mdp = V2.build_mdp(seed=1337)
    rng_theta = np.random.default_rng(7)
    theta = rng_theta.normal(scale=0.3, size=(mdp.n_states, mdp.n_actions))
    g_true = V2.true_policy_gradient(theta, mdp)

    print("=" * 78)
    print("B2 component ablation on the K=2 47.9x headline")
    print("=" * 78)
    print(f"||grad J||_2 = {float(np.linalg.norm(g_true)):.6f}")
    print(f"Delta grid = {DELTAS}, seeds = {SEEDS}, n_trials = {N_TRIALS}")
    print(f"MC per Delta = {N_TRIALS} x {len(SEEDS)} = {N_TRIALS*len(SEEDS)}")
    print("-" * 78)

    cells = {
        "full_RAC": RACConfig(
            tau_age=1000.0, is_clip=1.0, alpha_delta=1.0,
            max_correction_norm=1e9,
        ),
        "no_w_age": RACConfig(
            tau_age=1e12, is_clip=1.0, alpha_delta=1.0,
            max_correction_norm=1e9,
        ),
        "bare_additive": RACConfig(
            tau_age=1e12, is_clip=1e12, alpha_delta=1.0,
            max_correction_norm=1e9,
        ),
    }

    results = {}
    for label, cfg in cells.items():
        print(f"\n[{label}] tau_age={cfg.tau_age:g}, "
              f"is_clip={cfg.is_clip:g}, alpha={cfg.alpha_delta}")
        cell = run_cell(
            theta=theta, mdp=mdp, cfg=cfg, deltas=DELTAS,
            seeds=SEEDS, n_trials=N_TRIALS, T=T, g_true=g_true,
        )
        results[label] = cell
        print(f"  mean reduction = {cell['mean_reduction']:.2f}x  "
              f"(95% CI [{cell['ci95_lo']:.2f}, {cell['ci95_hi']:.2f}])")
        for delta, m in cell["per_delta"].items():
            print(f"  Delta={delta}: {m['reduction_factor']:.2f}x  "
                  f"VIF={m['vif']:.2f}")

    out_dir = ROOT / "results" / "track2_b2_ablation_47_9"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "config": dict(
            mdp_seed=1337, theta_seed=7,
            deltas=DELTAS, seeds=SEEDS,
            n_trials=N_TRIALS, T=T,
            actor="IdentityActor (rho=1 exactly)",
            cells={k: dict(
                tau_age=v.tau_age, is_clip=v.is_clip,
                alpha_delta=v.alpha_delta,
            ) for k, v in cells.items()},
        ),
        "results": results,
        "runtime_sec": time.time() - t0,
    }
    json_path = out_dir / "ablation.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o:
                  float(o) if isinstance(o, (np.floating,)) else
                  int(o) if isinstance(o, (np.integer,)) else
                  o.tolist() if isinstance(o, np.ndarray) else
                  (_ for _ in ()).throw(TypeError(str(type(o)))))
    print("\n" + "=" * 78)
    print(f"Wrote {json_path}")
    print(f"Total runtime: {time.time()-t0:.1f}s")
    print("=" * 78)
    print("\nTab 1 ablation rows (mean across Delta in {5,20,50}):")
    for label, cell in results.items():
        print(f"  {label:<16} {cell['mean_reduction']:>6.2f}x  "
              f"[{cell['ci95_lo']:>5.2f}, {cell['ci95_hi']:>5.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
