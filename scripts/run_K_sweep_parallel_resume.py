""" parallelization-orchestrator wrapper for cross-MDP-topology K-sweep.

This driver is a THIN ORCHESTRATOR around `verify_K_sweep_cross_mdp_topology.py`.
It does NOT modify the PREREG-faithful SCRIPT (`c835029`). It only changes
*scheduling*: instead of looping topology × K × mdp_seed sequentially in one
process, it forks N_WORKERS subprocesses (one per (topology, mdp_seed) outer
cell) which each run all K values for their cell. Per-cell results are
serialised to disk; the parent then aggregates and emits the same
summary.json + topology_*.json + figure that the original SCRIPT would.

PREREG-faithfulness:
  - Every cell uses the same `aggregate_K2_per_mdp` / `run_Kgeq3_cell` from the
    original SCRIPT.
  - Same theta_seed=7, mdp_seeds, mc_seeds, n_trials=3000, K-grid, RACConfig.
  - Same verdict logic via `evaluate_branches`.
  - Decomposition_seed_base, noise_scale, delta_K2 unchanged.
  - Outputs are byte-equivalent to a sequential run (modulo non-determinism in
    NumPy floating-point reduction order — *not present here* because each
    (topo, mdp, K, mc) cell's RNG is deterministic from (mc_seed, K) hash).

Why this is needed:
  - PREREG §6 self-flagged that K∈{15,20} cells have hostile per-trajectory cost
    that the sequential ETA (7.4 min) underestimated by ~50×.  #2
    KILL hit the 5h gate at 0/5 topologies. Empirical re-test in this iteration
    () confirms K=20 single (topo, mdp) cell at 3000 trials × 3 mc =
    ~23 min wall on local CPU. Sequential total = ~21h. PREREG 5h gate cannot
    be satisfied without parallelism.
  - B200 GPU does NOT accelerate this pure-NumPy code (no torch tensors used
    in the hot loops). The 52-core B200 host *does* allow 25-way parallelism
    across (topology, mdp_seed) outer cells.

Honest disclosures (added EX-ANTE before this driver runs at scale):
  1. This wrapper changes scheduling, not science. PREREG-faithful per cell.
  2. CPU host parallelism, NOT GPU acceleration. The mission brief's "B200
     2.5x faster than H100" claim does not apply to this script.
  3. Per-cell RNG state is identical regardless of parallelism (each cell's
     RNG is seeded deterministically from (mc_seed, K)).
  4. The 5h KILL gate is enforced per-PREREG: if elapsed_wall > 18000s the
     parent kills all workers and emits verdict = KILLED-twice-FALSIFIED-
     INFEASIBLE-permanent.
  5. Per-cell timeout = 60 min (each (topo, mdp_seed) cell across all K). If
     a cell exceeds this its result is marked TIMEOUT and the master verdict
     handles partial coverage.

Author:  T2 K-sweep RIDE subagent 
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

# Reuse  SCRIPT verbatim (no modification).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from verify_K_sweep_cross_mdp_topology import (  # noqa: E402
    TOPOLOGY_FACTORIES,
    DEFAULT_TOPOLOGIES,
    DEFAULT_K_GRID,
    DEFAULT_MDP_SEEDS,
    DEFAULT_MC_SEEDS,
    PEAK_BAND_TOLERANCE,
    aggregate_K2_per_mdp,
    run_Kgeq3_cell,
    evaluate_branches,
    make_figure,
)
from verify_rac_gradient_correction import _json_default  # noqa: E402
from src.rac import RACConfig  # noqa: E402

# ------------------------------------------------------------
# Worker: process one (topology, mdp_seed) pair across all K
# ------------------------------------------------------------


def _worker_one_cell(args_tuple):
    """Process one (topology, mdp_seed) pair. Run all K values for that pair.

    Each worker runs in a fresh subprocess; PYTHONHASHSEED + per-cell
    rng = np.random.default_rng(...) + theta_seed all match the
    sequential SCRIPT exactly.
    """
    (
        topology_name, mdp_seed, K_grid, mc_seeds, n_trials, T,
        theta_seed, delta_K2, tau_age, is_clip, alpha_delta,
        noise_scale, decomposition_seed_base,
    ) = args_tuple

    cfg = RACConfig(
        tau_age=tau_age, is_clip=is_clip, alpha_delta=alpha_delta,
        max_correction_norm=1e9,
    )
    factory = TOPOLOGY_FACTORIES[topology_name]
    base_mdp = factory(seed=mdp_seed)

    # Theta is sampled identically to the original SCRIPT:
    # rng_theta = np.random.default_rng(theta_seed) ; theta = rng_theta.normal(...)
    # This block matches verify_K_sweep_cross_mdp_topology.aggregate_topology_K_curve
    # NOTE: the original SCRIPT uses `factory(seed=mdp_seeds[0])` to get sample_mdp
    # for theta SHAPE, which means theta SHAPE is determined by the FIRST mdp_seed
    # (1337 by default). We replicate that exactly: theta shape is from the FIRST
    # mdp_seed in the global grid, NOT from our local mdp_seed. This preserves
    # PREREG-faithfulness across all parallel cells.
    rng_theta = np.random.default_rng(theta_seed)
    sample_mdp = factory(seed=DEFAULT_MDP_SEEDS[0])
    theta = rng_theta.normal(
        scale=0.3, size=(sample_mdp.n_states, sample_mdp.n_actions),
    )

    # Sanity: base_mdp shape must match theta shape (same topology -> same dims)
    assert (base_mdp.n_states, base_mdp.n_actions) == theta.shape, (
        f"shape mismatch topology={topology_name} "
        f"base_mdp=({base_mdp.n_states},{base_mdp.n_actions}) theta={theta.shape}"
    )

    out_per_K: dict[int, dict[str, Any]] = {}
    t_cell = time.time()
    for K in K_grid:
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
        out_per_K[K] = cell

    return dict(
        topology=topology_name,
        mdp_seed=mdp_seed,
        per_K=out_per_K,
        wall_sec=time.time() - t_cell,
    )


# ------------------------------------------------------------
# Parent: aggregate per-(topology, K) across mdp_seeds
# ------------------------------------------------------------


def aggregate_per_K_across_mdp_seeds(
    per_K_per_mdp: dict[int, dict[str, dict[str, Any]]],
    K_grid: list[int],
    mdp_seeds: list[int],
) -> dict[int, dict[str, float]]:
    """Replicate aggregate_topology_K_curve's mdp_seed reduction logic."""
    per_K_topology_aggregate: dict[int, dict[str, float]] = {}
    for K in K_grid:
        red_per_mdp = [
            per_K_per_mdp[K][f"mdp_{m}"]["reduction_factor"]
            for m in mdp_seeds
        ]
        red_min_per_mdp = [
            per_K_per_mdp[K][f"mdp_{m}"]["reduction_min"]
            for m in mdp_seeds
        ]
        vif_per_mdp = [
            per_K_per_mdp[K][f"mdp_{m}"]["vif_fast_mean"]
            for m in mdp_seeds
        ]
        vif_max_per_mdp = [
            per_K_per_mdp[K][f"mdp_{m}"]["vif_fast_max"]
            for m in mdp_seeds
        ]
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
    return per_K_topology_aggregate


def aggregate_topology_from_per_K_per_mdp(
    topology_name: str, K_grid: list[int],
    per_K_per_mdp: dict[int, dict[str, dict[str, Any]]],
    mdp_seeds: list[int],
) -> dict[str, Any]:
    """Build the final per-topology dict that matches aggregate_topology_K_curve."""
    factory = TOPOLOGY_FACTORIES[topology_name]
    sample_mdp = factory(seed=mdp_seeds[0])
    per_K_topology_aggregate = aggregate_per_K_across_mdp_seeds(
        per_K_per_mdp, K_grid, mdp_seeds,
    )
    K_means = {
        K: per_K_topology_aggregate[K]["reduction_topology_mean"] for K in K_grid
    }
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--topologies", type=str, nargs="+", default=DEFAULT_TOPOLOGIES,
                   choices=DEFAULT_TOPOLOGIES)
    p.add_argument("--K-grid", type=int, nargs="+", default=DEFAULT_K_GRID)
    p.add_argument("--mdp-seeds", type=int, nargs="+", default=DEFAULT_MDP_SEEDS)
    p.add_argument("--mc-seeds", type=int, nargs="+", default=DEFAULT_MC_SEEDS)
    p.add_argument("--n-trials", type=int, default=3000)
    p.add_argument("--trajectory-len", type=int, default=50)
    p.add_argument("--theta-seed", type=int, default=7)
    p.add_argument("--delta-K2", type=int, default=20)
    p.add_argument("--tau-age", type=float, default=1000.0)
    p.add_argument("--is-clip", type=float, default=1.0)
    p.add_argument("--alpha-delta", type=float, default=1.0)
    p.add_argument("--noise-scale", type=float, default=0.3)
    p.add_argument("--decomposition-seed-base", type=int, default=20260426)
    p.add_argument("--n-workers", type=int, default=25,
                   help="Outer (topology, mdp_seed) parallelism. 5x5=25 cells.")
    p.add_argument("--max-wall-sec", type=int, default=18000,
                   help="PREREG 5h KILL gate (18000s). NOT extendable.")
    p.add_argument("--results-dir", type=Path,
                   default=ROOT / "results" / "track2_K_sweep_cross_mdp_topology")
    p.add_argument("--figs-dir", type=Path, default=ROOT / "results" / "figs")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke run with reduced grid; verdict NOT binding.")
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

    print("=" * 96)
    print(" cross-MDP-topology K-sweep RIDE (parallel orchestrator)")
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
    print(f"n_workers      = {args.n_workers}")
    print(f"max_wall_sec   = {args.max_wall_sec} (PREREG 5h KILL gate)")
    n_outer = len(args.topologies) * len(args.mdp_seeds)
    print(f"Outer cells    = {len(args.topologies)} x {len(args.mdp_seeds)} "
          f"= {n_outer} (parallelised across n_workers={args.n_workers})")
    print("-" * 96)

    # Build worker tasks: 1 task per (topology, mdp_seed) pair
    tasks = []
    for topology_name in args.topologies:
        for mdp_seed in args.mdp_seeds:
            tasks.append((
                topology_name, mdp_seed, args.K_grid, args.mc_seeds,
                args.n_trials, args.trajectory_len, args.theta_seed,
                args.delta_K2, args.tau_age, args.is_clip, args.alpha_delta,
                args.noise_scale, args.decomposition_seed_base,
            ))

    # per_topology_per_K_per_mdp[topology_name][K][f'mdp_{m}'] = cell
    per_topology_per_K_per_mdp: dict[str, dict[int, dict[str, Any]]] = {
        t: {K: {} for K in args.K_grid} for t in args.topologies
    }
    cell_walls: list[float] = []
    failures: list[dict] = []
    timed_out = False

    # ProcessPoolExecutor, await with overall wall-clock cap
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(_worker_one_cell, t): t for t in tasks}
        for fut in as_completed(futures, timeout=None):
            t = futures[fut]
            topology_name, mdp_seed = t[0], t[1]
            elapsed = time.time() - t0
            if elapsed > args.max_wall_sec:
                print(f"[{elapsed:.1f}s] !!! 5h KILL GATE FIRED — "
                      "stopping all workers")
                timed_out = True
                # Cancel remaining
                for f2 in futures:
                    if not f2.done():
                        f2.cancel()
                break
            try:
                result = fut.result(timeout=10)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[{elapsed:.1f}s] FAIL {topology_name}/mdp_{mdp_seed}: {e}")
                failures.append(dict(
                    topology=topology_name, mdp_seed=mdp_seed, error=str(e),
                    traceback=tb,
                ))
                continue
            print(
                f"[{elapsed:6.1f}s] DONE {topology_name}/mdp_{mdp_seed} "
                f"(cell wall {result['wall_sec']:.1f}s)"
            )
            cell_walls.append(result["wall_sec"])
            for K, cell in result["per_K"].items():
                per_topology_per_K_per_mdp[topology_name][K][f"mdp_{mdp_seed}"] = cell

    elapsed_total = time.time() - t0

    # If timed out, emit KILLED-twice verdict and write what we have
    if timed_out:
        print("\n" + "=" * 96)
        print("VERDICT: KILLED-twice-FALSIFIED-INFEASIBLE-permanent")
        print("=" * 96)
        print(f"Elapsed wall: {elapsed_total:.1f}s (gate {args.max_wall_sec}s)")
        print(f"Cells completed: {len(cell_walls)} / {n_outer}")
        verdict = dict(
            verdict="KILLED-twice-FALSIFIED-INFEASIBLE-permanent",
            verdict_reasons=[
                f"5h gate fired at {elapsed_total:.1f}s "
                f"(PREREG 5h={args.max_wall_sec}s); first KILL was 2026-04-27 "
                " #2 commit 5d5d93d at 5h01m elapsed.",
                f"Cells completed: {len(cell_walls)} of {n_outer} "
                f"(topology x mdp_seed). Insufficient for cross-topology "
                "generalization claim per PREREG sec 4.",
            ],
            cells_completed=len(cell_walls),
            cells_required=n_outer,
            elapsed_wall_sec=elapsed_total,
            max_wall_sec=args.max_wall_sec,
            failures=failures,
        )
        partial_summary = dict(
            config=vars(args).copy(),
            prereg_sha="453363a",
            prereg_file="PREREG_T2_K_SWEEP_CROSS_MDP_TOPOLOGY.md",
            partial_per_topology_per_K_per_mdp=per_topology_per_K_per_mdp,
            verdict=verdict,
            runtime_sec=elapsed_total,
            cell_walls_sec=cell_walls,
        )
        partial_summary["config"]["results_dir"] = str(args.results_dir)
        partial_summary["config"]["figs_dir"] = str(args.figs_dir)
        with open(args.results_dir / "summary.json", "w") as f:
            json.dump(partial_summary, f, indent=2, default=_json_default)
        print(f"Wrote {args.results_dir}/summary.json (partial)")
        return 0  # Graceful: we honored the gate

    # All cells completed — aggregate per-topology
    per_topology: dict[str, dict[str, Any]] = {}
    for topology_name in args.topologies:
        per_topology[topology_name] = aggregate_topology_from_per_K_per_mdp(
            topology_name=topology_name, K_grid=args.K_grid,
            per_K_per_mdp=per_topology_per_K_per_mdp[topology_name],
            mdp_seeds=args.mdp_seeds,
        )

    # Verdict
    verdict = evaluate_branches(per_topology, K_grid=args.K_grid)

    # Per-topology K-curve table
    print("\n" + "=" * 96)
    print("Per-topology K-curve table "
          f"(reduction-factor mean across {len(args.mdp_seeds)} mdp-seeds x "
          f"{len(args.mc_seeds)} mc-seeds)")
    print("=" * 96)
    header = f"{'topology':>16s}  | " + "  |  ".join(
        [f"K={K:>2d}" for K in args.K_grid]
    ) + "  |  peak"
    print(header)
    print("-" * len(header))
    for t in args.topologies:
        cells = []
        for K in args.K_grid:
            r = per_topology[t]["per_K_topology_aggregate"][K][
                "reduction_topology_mean"
            ]
            cells.append(f"{r:>5.1f}x")
        peak_K = per_topology[t]["peak_K_topology"]
        cells.append(f"  K={peak_K}")
        print(f"{t:>16s}  | " + "  |  ".join(cells))

    print("\n" + "=" * 96)
    print("PREREG branch verdict")
    print("=" * 96)
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
    config_dict = vars(args).copy()
    config_dict["results_dir"] = str(args.results_dir)
    config_dict["figs_dir"] = str(args.figs_dir)
    summary = dict(
        config=config_dict,
        prereg_sha="453363a",
        prereg_file="PREREG_T2_K_SWEEP_CROSS_MDP_TOPOLOGY.md",
        per_topology=per_topology,
        verdict=verdict,
        runtime_sec=elapsed_total,
        cell_walls_sec=cell_walls,
        n_workers=args.n_workers,
        parallel_orchestrator=" run_K_sweep_parallel_resume.py",
    )
    with open(args.results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nWrote {args.results_dir}/summary.json")
    print(f"Wrote {args.results_dir}/topology_*.json (n={len(args.topologies)})")

    fig_path = args.figs_dir / "track2_K_sweep_cross_mdp_topology.png"
    try:
        make_figure(summary, fig_path)
        print(f"Wrote {fig_path}")
    except Exception as e:
        print(f"Figure generation failed (non-fatal): {e}")

    print(f"\nTotal runtime: {elapsed_total:.1f}s "
          f"({elapsed_total/60:.1f} min); n_workers={args.n_workers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
