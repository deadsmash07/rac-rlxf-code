"""K=10 tensor-Λ multichannel RAC gradient validator (Theorem A3 stress test).

Extends the K-sweep to K=10 channels with delays Δ ∈ {0,1,...,9}. This is
the ceiling data point for §4.2's K-scaling table: beyond K=10 the per-
channel signal share (1/K) drops so low that single-trajectory MC noise
can dominate the channel's own deterministic contribution to ∇J.

Mathematical setup
------------------
Same 3-state × 2-action MDP. Reward split across K=10 channels with the
exact-channel-sum-to-r_true invariant: each channel carries (1/10) r_true
plus zero-mean noise drawn so the K-vector of noises sums to 0 cell-wise.

Δ per channel: Δ_fast = 0, Δ_k = k for k=1..9. Wall-clock delay horizon
D_max = 9, twice K=5's horizon.

Tensor-Λ: uniform Λ[k, k] = 1.0 for k=0..9. (U-i) holds identically.

Estimators: g_true, g_fast_only (1/10 signal), g_fast_half (fast + 2 non-
fast = 3/10 signal), g_rac_tL (all K=10). Independent-channel VIF
denominator = Σ_{k=1..9} var(fast+k alone).

See K=5 docstring for the full mathematical specification; this file only
bumps K=10 and deltas=[0..9].

Usage
-----
    python scripts/verify_tensor_lambda_multichannel_K10.py
        # default: K=10, uniform Λ, Δ = (0..9), 3000 trials × seed 123

Outputs
-------
    results/tensor_lambda_multichannel_K10/validation.json
    results/figs/tensor_lambda_multichannel_K10.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import verify_rac_gradient_correction as V2  # noqa: E402
from src.rac import (  # noqa: E402
    CachedRolloutBatch,
    RACConfig,
    RolloutCache,
    SlowReward,
    TensorLambda,
    apply_rac_correction,
)


@dataclass(frozen=True)
class TabularMDPK:
    n_states: int
    n_actions: int
    gamma: float
    P: np.ndarray
    r_channels: tuple

    @property
    def r_total(self) -> np.ndarray:
        return sum(self.r_channels)  # type: ignore[return-value]


def build_mdp_kchannel(seed: int = 1337,
                        noise_scale: float = 0.3,
                        n_channels: int = 10) -> TabularMDPK:
    mdp2 = V2.build_mdp(seed=seed)
    r_true = mdp2.r_fast + mdp2.r_slow

    rng = np.random.default_rng(seed + 1)
    n_s, n_a = mdp2.n_states, mdp2.n_actions

    noises_indep = [
        rng.normal(scale=noise_scale, size=(n_s, n_a))
        for _ in range(n_channels - 1)
    ]
    noise_last = -sum(noises_indep)
    noises = noises_indep + [noise_last]

    r_channels = tuple(
        r_true / n_channels + noises[k] for k in range(n_channels)
    )
    assert np.allclose(sum(r_channels), r_true, atol=1e-12)

    return TabularMDPK(
        n_states=n_s, n_actions=n_a, gamma=mdp2.gamma,
        P=mdp2.P, r_channels=r_channels,
    )


def _as_tabular2(mdp: TabularMDPK, reward: np.ndarray) -> V2.TabularMDP:
    return V2.TabularMDP(
        n_states=mdp.n_states, n_actions=mdp.n_actions, gamma=mdp.gamma,
        P=mdp.P, r_fast=reward, r_slow=np.zeros_like(reward),
    )


def true_policy_gradient_k(theta: np.ndarray, mdp: TabularMDPK) -> np.ndarray:
    packed = _as_tabular2(mdp, mdp.r_total)
    return V2.true_policy_gradient(theta, packed)


def sample_trajectory_k(
    theta: np.ndarray,
    mdp: TabularMDPK,
    T: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    K = len(mdp.r_channels)
    pi = V2.softmax_policy(theta)

    states = np.empty(T, dtype=np.int64)
    actions = np.empty(T, dtype=np.int64)
    r_per_channel = [np.empty(T, dtype=np.float64) for _ in range(K)]

    s = int(rng.integers(0, mdp.n_states))
    for t in range(T):
        a = int(rng.choice(mdp.n_actions, p=pi[s, :]))
        states[t] = s
        actions[t] = a
        for k in range(K):
            r_per_channel[k][t] = mdp.r_channels[k][s, a]
        s = int(rng.choice(mdp.n_states, p=mdp.P[s, a, :]))

    G_per_channel = [np.empty(T, dtype=np.float64) for _ in range(K)]
    running = [0.0] * K
    for t in range(T - 1, -1, -1):
        for k in range(K):
            running[k] = r_per_channel[k][t] + mdp.gamma * running[k]
            G_per_channel[k][t] = running[k]

    out = dict(states=states, actions=actions)
    for k in range(K):
        out[f"r_{k}"] = r_per_channel[k]
        out[f"G_{k}"] = G_per_channel[k]
    return out


def _score_weighted_sum(
    advantages: np.ndarray, states: np.ndarray, actions: np.ndarray,
    theta: np.ndarray, mdp: TabularMDPK,
) -> np.ndarray:
    pi = V2.softmax_policy(theta)
    g = np.zeros_like(theta)
    for t in range(advantages.shape[0]):
        s, a = int(states[t]), int(actions[t])
        g += advantages[t] * V2.grad_log_pi_tabular(
            s, a, pi, mdp.n_states, mdp.n_actions,
        )
    return g / advantages.shape[0]


def naive_fast_only_pg(trajectory: dict[str, np.ndarray],
                        theta: np.ndarray,
                        mdp: TabularMDPK) -> np.ndarray:
    return _score_weighted_sum(
        trajectory["G_0"], trajectory["states"], trajectory["actions"],
        theta, mdp,
    )


def partial_fast_half_pg(trajectory: dict[str, np.ndarray],
                          theta: np.ndarray,
                          mdp: TabularMDPK,
                          k_max: int = 2) -> np.ndarray:
    K = len(mdp.r_channels)
    k_max = min(k_max, K - 1)
    partial = sum(trajectory[f"G_{k}"] for k in range(k_max + 1))
    return _score_weighted_sum(
        partial, trajectory["states"], trajectory["actions"], theta, mdp,
    )


def _build_cache_and_batch(
    trajectory: dict[str, np.ndarray],
    theta: np.ndarray,
    cache_ttl: int,
) -> tuple[RolloutCache, SimpleNamespace, int]:
    pi = V2.softmax_policy(theta)
    T = trajectory["states"].shape[0]
    rollout_step = 0
    log_pi_vec = np.log(pi[trajectory["states"], trajectory["actions"]] + 1e-30)

    cached = CachedRolloutBatch(
        step=rollout_step,
        uids=[str(t) for t in range(T)],
        prompt_ids=torch.zeros(T, 1, dtype=torch.long),
        response_ids=torch.zeros(T, 1, dtype=torch.long),
        attention_mask=torch.ones(T, 2, dtype=torch.long),
        response_mask=torch.ones(T, 1),
        old_log_probs=torch.tensor(log_pi_vec, dtype=torch.float64).unsqueeze(-1),
        A_partial=torch.tensor(trajectory["G_0"], dtype=torch.float64).unsqueeze(-1),
        r_fast=torch.tensor(trajectory["r_0"], dtype=torch.float64),
    )
    cache = RolloutCache(max_ttl_steps=cache_ttl + 10)
    cache.stash(rollout_step, cached)

    advantages = torch.tensor(
        trajectory["G_0"], dtype=torch.float64,
    ).unsqueeze(-1).clone()
    batch = SimpleNamespace(
        batch={"advantages": advantages},
        non_tensor_batch={
            "uid": np.array([str(t) for t in range(T)], dtype=object),
        },
        meta_info={"global_step": rollout_step + cache_ttl},
    )
    return cache, batch, rollout_step


def _matured_for_channel(
    trajectory: dict[str, np.ndarray],
    channel_name: str,
    G_channel: np.ndarray,
    weight: float,
    rollout_step: int,
) -> list[SlowReward]:
    T = G_channel.shape[0]
    return [
        SlowReward(
            uid=str(t), step_t=rollout_step,
            r_slow=torch.tensor([weight * G_channel[t]], dtype=torch.float64),
            channel_name=channel_name,
            fast_baseline=0.0,
        )
        for t in range(T)
    ]


def tensor_lambda_rac_pg_k(
    trajectory: dict[str, np.ndarray],
    theta: np.ndarray,
    mdp: TabularMDPK,
    Lambda: TensorLambda,
    delta_map: dict[int, list[tuple[int, float]]],
    cfg: RACConfig,
) -> np.ndarray:
    Lambda.validate()
    K = len(mdp.r_channels)
    max_delta = max(
        (d for k in range(1, K) for (d, _) in delta_map.get(k, [])),
        default=1,
    )
    cache, batch, rollout_step = _build_cache_and_batch(
        trajectory, theta, cache_ttl=max_delta,
    )

    channel_name_by_k = {k: f"channel_{k}" for k in range(1, K)}
    G_by_k = {k: trajectory[f"G_{k}"] for k in range(1, K)}
    for k in range(1, K):
        for (_delta, weight) in delta_map.get(k, []):
            matured_k = _matured_for_channel(
                trajectory=trajectory,
                channel_name=channel_name_by_k[k],
                G_channel=G_by_k[k],
                weight=weight,
                rollout_step=rollout_step,
            )
            apply_rac_correction(
                batch=batch,
                matured=matured_k,
                actor_rollout_wg=V2.IdentityActor(),
                rollout_cache=cache,
                cfg=cfg,
            )

    adv_corrected = batch.batch["advantages"].squeeze(-1).numpy()
    return _score_weighted_sum(
        adv_corrected, trajectory["states"], trajectory["actions"],
        theta, mdp,
    )


def per_channel_rac_pg_k(
    trajectory: dict[str, np.ndarray],
    theta: np.ndarray,
    mdp: TabularMDPK,
    delta_per_channel: dict[int, int],
    cfg: RACConfig,
) -> list[np.ndarray]:
    K = len(mdp.r_channels)
    results: list[np.ndarray] = []
    for k in range(1, K):
        delta_k = delta_per_channel[k]
        cache_k, batch_k, step = _build_cache_and_batch(
            trajectory, theta, cache_ttl=delta_k,
        )
        matured_k = _matured_for_channel(
            trajectory=trajectory, channel_name=f"channel_{k}",
            G_channel=trajectory[f"G_{k}"], weight=1.0, rollout_step=step,
        )
        apply_rac_correction(
            batch=batch_k, matured=matured_k,
            actor_rollout_wg=V2.IdentityActor(),
            rollout_cache=cache_k, cfg=cfg,
        )
        g_k = _score_weighted_sum(
            batch_k.batch["advantages"].squeeze(-1).numpy(),
            trajectory["states"], trajectory["actions"], theta, mdp,
        )
        results.append(g_k)
    return results


def make_lambda_uniform_k(
    deltas: list[int], D_max: int,
) -> tuple[TensorLambda, dict[int, list]]:
    K = len(deltas)
    L = TensorLambda(n_channels=K, D_max=D_max)
    delta_map: dict[int, list[tuple[int, float]]] = {}
    for k in range(K):
        L.set(k, deltas[k], 1.0)
        delta_map[k] = [(deltas[k], 1.0)]
    L.validate()
    return L, delta_map


def run_mc_cell(
    theta: np.ndarray, mdp: TabularMDPK, T: int,
    n_trials: int, delta_map: dict[int, list],
    Lambda: TensorLambda,
    seed: int, cfg: RACConfig,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed * 10_007 + 31)
    K = len(mdp.r_channels)
    delta_per_channel = {
        k: delta_map.get(k, [(1, 1.0)])[0][0] for k in range(1, K)
    }

    gs_fast = np.empty((n_trials, mdp.n_states, mdp.n_actions))
    gs_half = np.empty((n_trials, mdp.n_states, mdp.n_actions))
    gs_tL = np.empty((n_trials, mdp.n_states, mdp.n_actions))
    gs_indep = [
        np.empty((n_trials, mdp.n_states, mdp.n_actions))
        for _ in range(K - 1)
    ]

    for i in range(n_trials):
        traj = sample_trajectory_k(theta, mdp, T, rng)
        gs_fast[i] = naive_fast_only_pg(traj, theta, mdp)
        gs_half[i] = partial_fast_half_pg(traj, theta, mdp, k_max=2)
        gs_tL[i] = tensor_lambda_rac_pg_k(
            traj, theta, mdp, Lambda, delta_map, cfg,
        )
        per_channel = per_channel_rac_pg_k(
            traj, theta, mdp, delta_per_channel, cfg,
        )
        for j, g_j in enumerate(per_channel):
            gs_indep[j][i] = g_j

    out = dict(g_fast=gs_fast, g_half=gs_half, g_tL=gs_tL)
    for j, g_arr in enumerate(gs_indep):
        out[f"g_indep_{j+1}"] = g_arr
    return out


def summarize_cell(
    samples: dict[str, np.ndarray], g_true: np.ndarray, K: int,
) -> dict[str, Any]:
    def bias(arr: np.ndarray) -> float:
        return float(np.linalg.norm(arr.mean(axis=0) - g_true))

    def var(arr: np.ndarray) -> float:
        return float(arr.var(axis=0, ddof=1).mean())

    bias_fast = bias(samples["g_fast"])
    bias_half = bias(samples["g_half"])
    bias_tL = bias(samples["g_tL"])

    var_fast = var(samples["g_fast"])
    var_half = var(samples["g_half"])
    var_tL = var(samples["g_tL"])
    var_indep_sum = sum(
        var(samples[f"g_indep_{j}"]) for j in range(1, K)
    )

    return dict(
        bias_fast=bias_fast,
        bias_half=bias_half,
        bias_tL=bias_tL,
        var_fast=var_fast,
        var_half=var_half,
        var_tL=var_tL,
        var_indep_sum=var_indep_sum,
        reduction_factor_fast=bias_fast / max(bias_tL, 1e-12),
        reduction_factor_half=bias_half / max(bias_tL, 1e-12),
        vif_vs_fast=var_tL / max(var_fast, 1e-12),
        vif_vs_indep_sum=var_tL / max(var_indep_sum, 1e-12),
    )


def pool_across_seeds(seed_runs: list[dict[str, np.ndarray]],
                       g_true: np.ndarray, K: int) -> dict[str, Any]:
    pooled = {
        k: np.concatenate([r[k] for r in seed_runs], axis=0)
        for k in seed_runs[0].keys()
    }
    out = summarize_cell(pooled, g_true, K)
    out["per_seed_bias_tL"] = [
        float(np.linalg.norm(r["g_tL"].mean(axis=0) - g_true))
        for r in seed_runs
    ]
    out["per_seed_bias_fast"] = [
        float(np.linalg.norm(r["g_fast"].mean(axis=0) - g_true))
        for r in seed_runs
    ]
    return out


def make_figure(metrics: dict[str, Any], K: int, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=150)

    labels = [f"fast only\n(1/{K} signal)",
              f"fast+2 chans\n(3/{K} signal)",
              f"tensor-Λ RAC\n(K={K})"]
    biases = [metrics["bias_fast"], metrics["bias_half"], metrics["bias_tL"]]
    vars_ = [metrics["var_fast"], metrics["var_half"], metrics["var_tL"]]
    colors = ["#d62728", "#ff7f0e", "#17becf"]

    ax = axes[0]
    ax.bar(labels, biases, color=colors)
    ax.set_ylabel(r"$\|\mathrm{E}[\hat{g}] - \nabla J\|_2$")
    ax.set_title(f"Panel A — PG bias (K={K} channels)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(labels, vars_, color=colors)
    ax.set_ylabel("mean coord. variance")
    ax.set_title("Panel B — PG variance")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    per_seed = metrics["per_seed_bias_tL"]
    ax.plot(range(len(per_seed)), per_seed, "o-", color="#17becf",
            label="tensor-Λ RAC per-seed")
    ax.axhline(metrics["bias_tL"], color="gray", ls="--", alpha=0.5,
                label="pooled")
    ax.set_xlabel("seed idx")
    ax.set_ylabel("bias per seed")
    ax.set_title(f"Panel C — MC noise floor (K={K})")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Track 2 RAC tensor-Λ multichannel validator (K={K}, 3-state MDP)",
        fontsize=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


K_DEFAULT = 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--n-channels", type=int, default=K_DEFAULT)
    p.add_argument("--deltas", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    p.add_argument("--n-trials", type=int, default=3000)
    p.add_argument("--trajectory-len", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[123])
    p.add_argument("--mdp-seed", type=int, default=1337)
    p.add_argument("--theta-seed", type=int, default=7)
    p.add_argument("--noise-scale", type=float, default=0.3)
    p.add_argument("--tau-age", type=float, default=1000.0)
    p.add_argument("--is-clip", type=float, default=1.0)
    p.add_argument("--alpha-delta", type=float, default=1.0)
    p.add_argument("--reduction-gate", type=float, default=10.0)
    p.add_argument("--vif-gate-indep", type=float, default=1.5)
    p.add_argument(
        "--results-dir", type=Path,
        default=ROOT / "results" / f"tensor_lambda_multichannel_K{K_DEFAULT}",
    )
    p.add_argument("--figs-dir", type=Path, default=ROOT / "results" / "figs")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    if len(args.deltas) != args.n_channels:
        print(f"--deltas must have length {args.n_channels} (got "
              f"{len(args.deltas)}).", file=sys.stderr)
        return 2
    if args.deltas[0] != 0:
        print("WARNING: channel 0 Δ≠0.", file=sys.stderr)

    mdp = build_mdp_kchannel(
        seed=args.mdp_seed,
        noise_scale=args.noise_scale,
        n_channels=args.n_channels,
    )
    rng_theta = np.random.default_rng(args.theta_seed)
    theta = rng_theta.normal(scale=0.3, size=(mdp.n_states, mdp.n_actions))

    cfg = RACConfig(
        tau_age=args.tau_age, is_clip=args.is_clip,
        alpha_delta=args.alpha_delta, max_correction_norm=1e9,
    )
    D_max = max(args.deltas) + 5
    Lambda, delta_map = make_lambda_uniform_k(
        deltas=args.deltas, D_max=D_max,
    )
    g_true = true_policy_gradient_k(theta, mdp)

    print("=" * 78)
    print(f"K={args.n_channels} tensor-Λ multichannel RAC validator (Track 2, 3-state MDP)")
    print("=" * 78)
    print(f"|S|={mdp.n_states}, |A|={mdp.n_actions}, γ={mdp.gamma}")
    print(f"||∇J(θ)||₂ = {float(np.linalg.norm(g_true)):.6f}")
    print(f"Λ config   = uniform ({args.n_channels} channels, one Δ each)")
    print(f"deltas     = {args.deltas}")
    print(
        "(U-i) sums: "
        + ", ".join(
            f"k={k}→{Lambda.per_channel_sum(k):.2f}"
            for k in range(Lambda.n_channels)
        )
    )
    print(f"n_trials × n_seeds = {args.n_trials} × {len(args.seeds)} = "
          f"{args.n_trials * len(args.seeds)}")
    print(f"RAC cfg    = tau_age={cfg.tau_age}, is_clip={cfg.is_clip}, "
          f"α={cfg.alpha_delta}")
    print("-" * 78)

    seed_runs: list[dict[str, np.ndarray]] = []
    for seed in args.seeds:
        seed_runs.append(run_mc_cell(
            theta=theta, mdp=mdp, T=args.trajectory_len,
            n_trials=args.n_trials, delta_map=delta_map, Lambda=Lambda,
            seed=seed, cfg=cfg,
        ))
    metrics = pool_across_seeds(seed_runs, g_true, K=args.n_channels)

    print(
        f"bias_fast  = {metrics['bias_fast']:.4f}  "
        f"bias_half  = {metrics['bias_half']:.4f}  "
        f"bias_tL    = {metrics['bias_tL']:.4f}"
    )
    print(
        f"var_fast   = {metrics['var_fast']:.4f}  "
        f"var_half   = {metrics['var_half']:.4f}  "
        f"var_tL     = {metrics['var_tL']:.4f}  "
        f"var_indep_sum = {metrics['var_indep_sum']:.4f}"
    )
    print(
        f"reduction × vs fast        = {metrics['reduction_factor_fast']:.2f}×\n"
        f"reduction × vs fast+half   = {metrics['reduction_factor_half']:.2f}×\n"
        f"VIF vs fast-only var       = {metrics['vif_vs_fast']:.2f}×\n"
        f"VIF vs indep-sum var       = {metrics['vif_vs_indep_sum']:.2f}×"
    )
    print(f"per-seed bias_tL: {['%.4f' % b for b in metrics['per_seed_bias_tL']]}")

    bias_pass = metrics["reduction_factor_fast"] >= args.reduction_gate
    vif_pass = metrics["vif_vs_indep_sum"] <= args.vif_gate_indep
    overall = bias_pass and vif_pass

    print("-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"  BIAS_REDUCTION ≥ {args.reduction_gate}×: "
          f"{'PASS' if bias_pass else 'FAIL'} "
          f"({metrics['reduction_factor_fast']:.2f}×)")
    print(f"  VIF_INDEP ≤ {args.vif_gate_indep}×      : "
          f"{'PASS' if vif_pass else 'FAIL'} "
          f"({metrics['vif_vs_indep_sum']:.2f}×)")
    print(f"  OVERALL:            {'PASS' if overall else 'FAIL'}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figs_dir.mkdir(parents=True, exist_ok=True)

    out = dict(
        mdp=dict(
            n_states=mdp.n_states, n_actions=mdp.n_actions, gamma=mdp.gamma,
            r_channels=[r.tolist() for r in mdp.r_channels],
            P=mdp.P.tolist(),
        ),
        theta=theta.tolist(),
        grad_true_l2=float(np.linalg.norm(g_true)),
        grad_true=g_true.tolist(),
        config=dict(
            n_channels=args.n_channels,
            lambda_config="uniform",
            deltas=args.deltas,
            n_trials=args.n_trials,
            trajectory_len=args.trajectory_len,
            seeds=args.seeds,
            rac_cfg=asdict(cfg),
            reduction_gate=args.reduction_gate,
            vif_gate_indep=args.vif_gate_indep,
            noise_scale=args.noise_scale,
        ),
        per_channel_sums=[
            float(Lambda.per_channel_sum(k)) for k in range(Lambda.n_channels)
        ],
        metrics=metrics,
        overall_pass=bool(overall),
        runtime_sec=time.time() - t0,
    )
    json_path = args.results_dir / "validation.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=V2._json_default)
    print(f"Wrote {json_path}")

    fig_path = args.figs_dir / f"tensor_lambda_multichannel_K{args.n_channels}.png"
    make_figure(metrics, args.n_channels, fig_path)
    print(f"Wrote {fig_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
