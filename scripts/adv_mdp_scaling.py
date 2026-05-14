"""EXP-C — MDP-size scaling of the RAC K=2 bias-reduction.

Reviewer attack class targeted: "the 47.9x bias-reduction is an
artifact of the 3x2 toy. Does it transfer to larger tabular MDPs?"

Setup
-----
We construct a family of canonical-style tabular MDPs with sizes
(n_states, n_actions) in {(3,2), (5,3), (10,5), (20,8)}, holding the
construction recipe identical:

  - P[s,a,:] = softmax of standard-normal logits (full-support kernel).
  - r_fast[s,a] = Uniform(-0.5, 0.5).
  - r_slow[s,a] = (small structured bump on the maximal slot) + state-
    dependent Uniform(-0.3, 0.3) jitter.
  - gamma = 0.9.

We use the 47.9x canonical RAC configuration
(tau_age=1000, is_clip=1.0, alpha_delta=1.0, IdentityActor rho=1) and
the K=2 working point. Each MDP size is exercised at deltas in
{5, 20, 50}, 5 MDP seeds (so the bias-reduction averages over the
MDP-structure ensemble at each size), and 1000 trials per cell.

Outputs
-------
- results/adv_mdp_scaling_<timestamp>/results.json

Runtime: ~30-90 min on a single Mac CPU core
(4 sizes x 3 deltas x 5 seeds x 1000 trials = 60k trajectories on
the smallest MDP; 20x8 closed-form Bellman is 160x160 linear solve
per trial which dominates).

Skill citation: research-paper-adversarial-review-icml-neurips
§stress-tests; professional-rl-reviewer §robustness-ablation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import verify_rac_gradient_correction as V2  # noqa: E402
from src.rac import RACConfig  # noqa: E402


def build_mdp_sized(
    n_states: int,
    n_actions: int,
    seed: int = 1337,
    gamma: float = 0.9,
) -> V2.TabularMDP:
    """Generalisation of `verify_rac_gradient_correction.build_mdp` to (n_s, n_a).

    Same recipe: softmax transitions over n_s next-states, fast reward
    Uniform(-0.5, 0.5), slow reward = small structured bump (on the cell
    [0, n_actions-1] — i.e. action `n_actions-1` in state 0) plus
    state-dependent jitter.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n_states, n_actions, n_states))
    P = np.exp(logits - logits.max(axis=-1, keepdims=True))
    P = P / P.sum(axis=-1, keepdims=True)

    r_fast = rng.uniform(-0.5, 0.5, size=(n_states, n_actions))
    r_slow = np.zeros((n_states, n_actions))
    # Structured bump on the (0, last-action) cell, matching the
    # canonical 3x2 recipe's r_slow[0, 1] += 0.6.
    r_slow[0, n_actions - 1] += 0.6
    r_slow += rng.uniform(-0.3, 0.3, size=(n_states, n_actions))

    return V2.TabularMDP(
        n_states=n_states, n_actions=n_actions, gamma=gamma,
        P=P, r_fast=r_fast, r_slow=r_slow,
    )


def run_size_cell(
    theta: np.ndarray,
    mdp: V2.TabularMDP,
    cfg: RACConfig,
    delta_steps: int,
    seed: int,
    n_trials: int,
    T: int,
) -> dict[str, np.ndarray]:
    """One Monte-Carlo cell at (size, delta, seed)."""
    rng = np.random.default_rng(
        seed * 10_007 + delta_steps * 97 + mdp.n_states * 31 + mdp.n_actions,
    )
    n_s, n_a = mdp.n_states, mdp.n_actions
    g_naive_all = np.empty((n_trials, n_s, n_a))
    g_rac_all = np.empty((n_trials, n_s, n_a))
    for i in range(n_trials):
        traj = V2.sample_trajectory(theta, mdp, T, rng)
        g_naive_all[i] = V2.naive_pg_estimator(traj, theta, mdp)
        g_rac_all[i] = V2.rac_corrected_pg_estimator(
            traj, theta, mdp, delta_steps, cfg,
        )
    return dict(g_naive=g_naive_all, g_rac=g_rac_all)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--sizes", type=str, nargs="+",
        default=["3x2", "5x3", "10x5", "20x8"],
        help="MDP sizes as 'n_states x n_actions' tokens.",
    )
    p.add_argument("--deltas", type=int, nargs="+", default=[5, 20, 50])
    p.add_argument(
        "--mdp-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="MDP-structure seeds — each size is averaged over these.",
    )
    p.add_argument(
        "--theta-seed", type=int, default=7,
        help="Seed for the initial theta perturbation (held across MDPs).",
    )
    p.add_argument("--n-trials", type=int, default=1000)
    p.add_argument("--trajectory-len", type=int, default=50)
    p.add_argument("--tag", type=str, default=None)
    return p.parse_args()


def parse_size(token: str) -> tuple[int, int]:
    ns, na = token.lower().split("x")
    return int(ns), int(na)


def main() -> int:
    args = parse_args()
    t0 = time.time()
    tag = args.tag or time.strftime("%Y%m%d_%H%M")

    cfg = RACConfig(
        tau_age=1000.0, is_clip=1.0, alpha_delta=1.0,
        max_correction_norm=1e9,
    )

    sizes = [parse_size(s) for s in args.sizes]

    print("=" * 78)
    print("EXP-C: MDP-SIZE SCALING SWEEP")
    print("=" * 78)
    print(f"sizes     = {sizes}")
    print(f"deltas    = {args.deltas}")
    print(f"mdp_seeds = {args.mdp_seeds}")
    print(f"n_trials  = {args.n_trials}, T = {args.trajectory_len}")
    print(f"RAC cfg = tau_age={cfg.tau_age}, is_clip={cfg.is_clip}, "
          f"alpha={cfg.alpha_delta}")
    print("-" * 78)

    per_size: dict[str, Any] = {}
    for (n_s, n_a) in sizes:
        size_key = f"{n_s}x{n_a}"
        print(f"\n[size {size_key}]  ({n_s} states x {n_a} actions)")
        per_delta: dict[int, list[float]] = {d: [] for d in args.deltas}
        per_delta_full: dict[int, list[dict]] = {d: [] for d in args.deltas}
        for mdp_seed in args.mdp_seeds:
            mdp = build_mdp_sized(n_s, n_a, seed=mdp_seed + 1337)
            rng_theta = np.random.default_rng(args.theta_seed + mdp_seed)
            theta = rng_theta.normal(scale=0.3, size=(n_s, n_a))
            g_true = V2.true_policy_gradient(theta, mdp)
            for delta in args.deltas:
                cell = run_size_cell(
                    theta=theta, mdp=mdp, cfg=cfg,
                    delta_steps=delta, seed=mdp_seed,
                    n_trials=args.n_trials, T=args.trajectory_len,
                )
                metrics = V2.summarize_cell(cell["g_naive"], cell["g_rac"], g_true)
                per_delta[delta].append(metrics["reduction_factor"])
                per_delta_full[delta].append(metrics)
                print(
                    f"  mdp_seed={mdp_seed} Delta={delta:3d}  "
                    f"bias_naive={metrics['bias_naive']:.4f}  "
                    f"bias_rac={metrics['bias_rac']:.4f}  "
                    f"red x={metrics['reduction_factor']:>6.2f}  "
                    f"VIF={metrics['vif']:.2f}"
                )

        # Aggregate across delta x mdp_seed
        all_red = [r for d in args.deltas for r in per_delta[d]]
        per_size[size_key] = dict(
            n_states=n_s, n_actions=n_a,
            per_delta_per_seed_reduction={
                str(d): per_delta[d] for d in args.deltas
            },
            per_delta_full={
                str(d): per_delta_full[d] for d in args.deltas
            },
            mean_reduction=float(np.mean(all_red)),
            median_reduction=float(np.median(all_red)),
            min_reduction=float(np.min(all_red)),
            max_reduction=float(np.max(all_red)),
            std_reduction=float(np.std(all_red)),
        )
        c = per_size[size_key]
        print(f"  ---> mean = {c['mean_reduction']:.2f}x   "
              f"median = {c['median_reduction']:.2f}x   "
              f"min = {c['min_reduction']:.2f}x   "
              f"max = {c['max_reduction']:.2f}x   "
              f"std = {c['std_reduction']:.2f}")

    print("=" * 78)
    print("AGGREGATE BY SIZE:")
    for k, v in per_size.items():
        print(f"  {k:<8}  mean = {v['mean_reduction']:.2f}x   "
              f"median = {v['median_reduction']:.2f}x   "
              f"std = {v['std_reduction']:.2f}")

    out_dir = ROOT / "results" / f"adv_mdp_scaling_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = dict(
        experiment="EXP-C MDP-size scaling sweep",
        config=dict(
            sizes=[f"{ns}x{na}" for (ns, na) in sizes],
            deltas=list(args.deltas),
            mdp_seeds=list(args.mdp_seeds),
            theta_seed=args.theta_seed,
            n_trials=args.n_trials,
            trajectory_len=args.trajectory_len,
            rac_cfg=dict(tau_age=cfg.tau_age, is_clip=cfg.is_clip,
                         alpha_delta=cfg.alpha_delta),
        ),
        results=per_size,
        runtime_sec=time.time() - t0,
    )
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=V2._json_default)
    print(f"Wrote {json_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
