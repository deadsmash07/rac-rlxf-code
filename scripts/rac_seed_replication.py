"""Seed-replication of V-trace identity-collapse + Lambda-slack sweep.

Authoring metadata
------------------
- Author: iter+N+RLxF-resume-30 SCRIPT-author (Opus 4.7 1M-context, MAX 4D)
- Iteration: iter+N+RLxF-resume-30 (seed-robustness replication)
- Dispatch target: RunPod GPU A (H100 80GB; pod port 17321; host 103.207.149.65)
- Skill citation: professional-rl-reviewer Section replication-robustness +
  launch-runpod-h100-job Section dispatch.

What this script does
---------------------
Confirms that the existing seed=42 verdicts replicate at additional head seeds
{43, 44, 45}, satisfying the reviewer attack "your identity / slack results
are an artifact of the random head seed".

Both identities under test are algebraic (Theorem 1 special case at Lambda=I;
linear slack at constant Lambda<1), so the head seed should be irrelevant once
the Qwen backbone produces deterministic hidden states. We verify this
empirically rather than relying on the algebra alone.

Optimization
------------
A naive implementation would (gen, score r_slow, score r_fast) per seed, but
generation is greedy (do_sample=False) so it is deterministic across seeds,
and Skywork r_slow is seed-independent. We therefore:
  1. Generate N=500 responses ONCE.
  2. Score r_slow ONCE.
  3. For each seed in {43, 44, 45}: re-init the fast head, score r_fast,
     run identity-kernel check + slack-sweep algebra (pure numpy, seconds).

This collapses 3 x (gen + slow + fast) into 1 x gen + 1 x slow + 3 x fast,
saving ~25 min of wall-time on H100.

Outputs (under --output_dir, default results/rac_seed_replication/)
------------------------------------------------------------------
  identity_seed{43,44,45}.json      one per seed, mirrors rac_vtrace_identity
  slack_seed{43,44,45}.json         one per seed, mirrors rac_lambda_slack_sweep
  summary.json                       aggregate pass/fail verdict
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adv_quality_7B import (  # noqa: E402
    POLICY_MODEL_ID,
    FAST_RM_MODEL_ID,
    SLOW_RM_MODEL_ID,
    load_policy,
    load_fast_rm,
    load_slow_rm,
    generate_responses,
    make_fast_head,
    score_with_fast_rm,
    score_with_slow_rm,
)
from rac_vtrace_identity_kernel_check import (  # noqa: E402
    compute_baseline,
    advantage_vtrace_identity,
    advantage_rac_identity_kernel,
)
from rac_lambda_slack_check import (  # noqa: E402
    advantage_oracle,
    evaluate_kernel,
)


DEFAULT_SEEDS = (43, 44, 45)
DEFAULT_SLACK_ETAS = (0.05, 0.15, 0.30)


def run_identity_check(
    r_fast: np.ndarray, r_slow: np.ndarray, *,
    tau_age: float, rho_clip: float, threshold: float, head_seed: int,
) -> dict:
    r_fast64 = r_fast.astype(np.float64)
    r_slow64 = r_slow.astype(np.float64)
    baseline = compute_baseline(r_fast64)

    A_vtrace = advantage_vtrace_identity(r_slow64, baseline)
    A_rac = advantage_rac_identity_kernel(
        r_fast64, r_slow64, baseline,
        tau_age=tau_age, rho_clip=rho_clip,
    )
    diff = A_rac - A_vtrace
    max_abs_diff = float(np.max(np.abs(diff)))
    mean_abs_diff = float(np.mean(np.abs(diff)))
    passed = max_abs_diff < threshold
    verdict = (
        "ESTABLISHED-IDENTITY-AT-LLM-SCALE" if passed
        else "FALSIFIED-IDENTITY-AT-LLM-SCALE"
    )
    return {
        "head_seed": int(head_seed),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "l_inf": max_abs_diff,
        "threshold": float(threshold),
        "passed": bool(passed),
        "verdict_class": verdict,
        "first10_A_rac":    [float(v) for v in A_rac[:10]],
        "first10_A_vtrace": [float(v) for v in A_vtrace[:10]],
        "first10_diff":     [float(v) for v in diff[:10]],
    }


def run_slack_sweep(
    r_fast: np.ndarray, r_slow: np.ndarray, *,
    etas: tuple, tau_age: float, rho_clip: float, head_seed: int,
) -> dict:
    r_fast64 = r_fast.astype(np.float64)
    r_slow64 = r_slow.astype(np.float64)
    baseline = compute_baseline(r_fast64)
    A_oracle = advantage_oracle(r_slow64, baseline)

    per_eta = []
    for eta in etas:
        lambda_zero = 1.0 - float(eta)
        res = evaluate_kernel(
            r_fast64, r_slow64, baseline, A_oracle,
            lambda_zero=lambda_zero, tau_age=tau_age, rho_clip=rho_clip,
        )
        res["eta"] = float(eta)
        per_eta.append(res)
    return {
        "head_seed": int(head_seed),
        "etas": [float(e) for e in etas],
        "per_eta": per_eta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--gen_batch_size", type=int, default=4)
    ap.add_argument("--score_batch_size", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--slack_etas", type=float, nargs="+",
                    default=list(DEFAULT_SLACK_ETAS))
    ap.add_argument("--tau_age", type=float, default=1000.0)
    ap.add_argument("--rho_clip", type=float, default=1.0)
    ap.add_argument("--identity_threshold", type=float, default=1e-5)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[start] rac_seed_replication  n={args.n_prompts}  "
          f"seeds={args.seeds}  etas={args.slack_etas}", flush=True)
    print(f"[start] cuda_devices={torch.cuda.device_count()} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    t_start = time.time()

    from datasets import load_dataset
    print("[data] loading UltraFeedback test_prefs...", flush=True)
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized",
                      split="test_prefs")
    prompts = [ds[i]["prompt"] for i in range(min(args.n_prompts, len(ds)))]
    print(f"[data] loaded {len(prompts)} prompts", flush=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ----- Phase 1: generate (seed-independent, greedy) -----
    print("\n=== Phase 1: generate responses (greedy, seed-independent) ===",
          flush=True)
    pol, pol_tok = load_policy(bnb)
    responses = generate_responses(pol, pol_tok, prompts,
                                   max_new_tokens=args.max_new_tokens,
                                   batch_size=args.gen_batch_size)
    print(f"[gen] complete  responses={len(responses)}", flush=True)
    del pol
    torch.cuda.empty_cache()
    print(f"[mem] cuda_alloc={torch.cuda.memory_allocated() / 1e9:.2f} GB",
          flush=True)

    # ----- Phase 2: slow RM score (seed-independent) -----
    print("\n=== Phase 2: score with slow RM (Skywork; seed-independent) ===",
          flush=True)
    slow_rm, slow_tok = load_slow_rm(bnb)
    r_slow = score_with_slow_rm(slow_rm, slow_tok, prompts, responses,
                                batch_size=args.score_batch_size)
    print(f"[slow-score] mean={r_slow.mean():.4f}  std={r_slow.std():.4f}",
          flush=True)
    del slow_rm
    torch.cuda.empty_cache()
    print(f"[mem] cuda_alloc={torch.cuda.memory_allocated() / 1e9:.2f} GB",
          flush=True)

    # ----- Phase 3: fast RM, scored once-per-seed -----
    print("\n=== Phase 3: score with fast RM (Qwen2.5-7B + random head) ===",
          flush=True)
    fast_rm, fast_tok = load_fast_rm(bnb)
    hidden_size = int(fast_rm.config.hidden_size)
    fast_device = next(fast_rm.parameters()).device

    per_seed_identity = []
    per_seed_slack = []
    per_seed_diag = []

    for seed in args.seeds:
        print(f"\n--- seed={seed} ---", flush=True)
        head = make_fast_head(hidden_size, seed=int(seed),
                              device=fast_device, dtype=torch.bfloat16)
        r_fast = score_with_fast_rm(fast_rm, fast_tok, head, prompts, responses,
                                    batch_size=args.score_batch_size)
        print(f"[seed={seed}] r_fast mean={r_fast.mean():.4f} "
              f"std={r_fast.std():.4f}", flush=True)

        # Persist raw r_fast for downstream reproducibility.
        np.save(out_dir / f"r_fast_seed{seed}.npy", r_fast)

        # Identity-kernel check.
        id_res = run_identity_check(
            r_fast, r_slow,
            tau_age=args.tau_age, rho_clip=args.rho_clip,
            threshold=args.identity_threshold, head_seed=int(seed),
        )
        id_payload = {
            "config": {
                "n_prompts": args.n_prompts,
                "max_new_tokens": args.max_new_tokens,
                "head_seed": int(seed),
                "tau_age": args.tau_age,
                "rho_clip": args.rho_clip,
                "identity_threshold": float(args.identity_threshold),
                "policy_model": POLICY_MODEL_ID,
                "fast_rm": FAST_RM_MODEL_ID,
                "slow_rm": SLOW_RM_MODEL_ID,
            },
            "identity_check": id_res,
            "diagnostics": {
                "r_fast_mean": float(r_fast.mean()),
                "r_fast_std":  float(r_fast.std()),
                "r_slow_mean": float(r_slow.mean()),
                "r_slow_std":  float(r_slow.std()),
            },
        }
        (out_dir / f"identity_seed{seed}.json").write_text(
            json.dumps(id_payload, indent=2))
        print(f"[seed={seed}] identity l_inf = {id_res['l_inf']:.3e}  "
              f"verdict={id_res['verdict_class']}", flush=True)

        # Slack-sweep across eta in {0.05, 0.15, 0.30}.
        slack_res = run_slack_sweep(
            r_fast, r_slow,
            etas=tuple(args.slack_etas),
            tau_age=args.tau_age, rho_clip=args.rho_clip,
            head_seed=int(seed),
        )
        slack_payload = {
            "config": {
                "n_prompts": args.n_prompts,
                "head_seed": int(seed),
                "tau_age": args.tau_age,
                "rho_clip": args.rho_clip,
                "etas": [float(e) for e in args.slack_etas],
                "policy_model": POLICY_MODEL_ID,
                "fast_rm": FAST_RM_MODEL_ID,
                "slow_rm": SLOW_RM_MODEL_ID,
            },
            "slack_sweep": slack_res,
            "diagnostics": {
                "r_fast_mean": float(r_fast.mean()),
                "r_fast_std":  float(r_fast.std()),
                "r_slow_mean": float(r_slow.mean()),
                "r_slow_std":  float(r_slow.std()),
                "residual_mean": float((r_slow - r_fast).mean()),
                "residual_abs_mean": float(np.abs(r_slow - r_fast).mean()),
            },
        }
        (out_dir / f"slack_seed{seed}.json").write_text(
            json.dumps(slack_payload, indent=2))
        for r in slack_res["per_eta"]:
            print(f"[seed={seed}] eta={r['eta']:.2f}  "
                  f"pointwise_ratio_mean={r['pointwise_ratio_mean']}  "
                  f"std={r['pointwise_ratio_std']}  "
                  f"verdict={r['verdict_class']}", flush=True)

        per_seed_identity.append(id_res)
        per_seed_slack.append(slack_res)
        per_seed_diag.append({
            "head_seed": int(seed),
            "r_fast_mean": float(r_fast.mean()),
            "r_fast_std": float(r_fast.std()),
        })

        del head
        torch.cuda.empty_cache()

    del fast_rm
    torch.cuda.empty_cache()

    # ----- Phase 4: aggregate summary -----
    print("\n=== Phase 4: aggregate summary ===", flush=True)

    all_identity_pass = all(r["passed"] for r in per_seed_identity)
    all_identity_zero = all(r["l_inf"] == 0.0 for r in per_seed_identity)

    # Slack: per (seed, eta) cell must be ratio = 1 +/- machine zero.
    slack_cells = []
    for seed_pack in per_seed_slack:
        seed = seed_pack["head_seed"]
        for r in seed_pack["per_eta"]:
            cell = {
                "head_seed": int(seed),
                "eta": float(r["eta"]),
                "pointwise_ratio_mean": r["pointwise_ratio_mean"],
                "pointwise_ratio_std": r["pointwise_ratio_std"],
                "passed": bool(
                    r["pointwise_ratio_mean"] is not None
                    and abs(r["pointwise_ratio_mean"] - 1.0) < 1e-6
                    and r["pointwise_ratio_std"] is not None
                    and r["pointwise_ratio_std"] < 1e-14
                ),
                "verdict_class": r["verdict_class"],
            }
            slack_cells.append(cell)
    all_slack_pass = all(c["passed"] for c in slack_cells)

    elapsed = time.time() - t_start
    summary = {
        "config": {
            "n_prompts": args.n_prompts,
            "seeds": [int(s) for s in args.seeds],
            "slack_etas": [float(e) for e in args.slack_etas],
            "tau_age": args.tau_age,
            "rho_clip": args.rho_clip,
            "identity_threshold": float(args.identity_threshold),
            "baseline_seed_in_paper": 42,
            "policy_model": POLICY_MODEL_ID,
            "fast_rm": FAST_RM_MODEL_ID,
            "slow_rm": SLOW_RM_MODEL_ID,
        },
        "identity_per_seed": [
            {
                "head_seed": r["head_seed"],
                "l_inf": r["l_inf"],
                "passed": r["passed"],
                "verdict_class": r["verdict_class"],
            }
            for r in per_seed_identity
        ],
        "slack_per_cell": slack_cells,
        "diagnostics_per_seed": per_seed_diag,
        "aggregate_verdict": {
            "all_identity_passed": bool(all_identity_pass),
            "all_identity_l_inf_zero": bool(all_identity_zero),
            "all_slack_passed_machine_zero": bool(all_slack_pass),
            "overall_pass": bool(all_identity_pass and all_slack_pass),
            "verdict_class": (
                "ESTABLISHED-REPLICATION-ACROSS-SEEDS"
                if (all_identity_pass and all_slack_pass)
                else "FAILED-REPLICATION-ACROSS-SEEDS"
            ),
        },
        "wall_time_seconds": elapsed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== SUMMARY ===")
    print(f"  seeds tested      = {args.seeds}")
    print(f"  identity l_inf=0  = {all_identity_zero}")
    print(f"  identity all PASS = {all_identity_pass}")
    print(f"  slack all PASS    = {all_slack_pass}")
    print(f"  OVERALL PASS      = {summary['aggregate_verdict']['overall_pass']}")
    print(f"  wall_time         = {elapsed:.0f}s")
    print(f"\n[done] wrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
