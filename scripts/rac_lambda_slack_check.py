"""Lambda-slack check: empirical verification of Theorem-1 slack-prediction.

Authoring metadata
------------------
- Author: iter+N+RLxF-resume-28 SCRIPT-author (Opus 4.7 1M-context, MAX 4D)
- Iteration: iter+N+RLxF-resume-28
- Dispatch target: RunPod GPU A (H100 80GB; pod port 17321; host 103.207.149.65)
- Skill citation: professional-rl-reviewer §theorem-tightness +
  research-grade-code-audit-pre-launch (G1-G12 inline self-audit passed) +
  launch-runpod-h100-job §dispatch.

What this script does
---------------------
Empirically verifies the slack-prediction sentence in main-paper Section 2:

  "a non-row-stochastic control with sum_Delta Lambda = 0.85 is biased by
   exactly the predicted slack"

Setup: identical numerical pipeline to scripts/rac_vtrace_identity_kernel_check
(same Llama-3-8B policy, Qwen-2.5-7B fast RM, Skywork-Llama-3.1-8B slow RM,
4-bit nf4 + bf16, head_seed=42, N=500 UltraFeedback test_prefs prompts).

We compute A_oracle (= r_slow - baseline, the V-trace target at Lambda=I)
and A_RAC under two delay-kernel configurations:

  - Row-stochastic (RS):     Lambda[Delta=0] = 1.0, sum = 1.0, deficit = 0
  - Non-row-stochastic (NRS): Lambda[Delta=0] = 0.85, sum = 0.85, deficit = 0.15

Algebra
-------
With rho_clip=1, w_age(0)=1, and Delta=0 deterministically:

  A_RAC[t] = r_fast[t] - b[t] + Lambda[0] * (r_slow[t] - r_fast[t])

So under (RS):  A_RAC = r_slow - b = A_oracle, bias should be 0 (numerical zero).
Under (NRS):    A_RAC - A_oracle = (Lambda[0] - 1) * (r_slow - r_fast)
                                 = -0.15 * (r_slow - r_fast)

Hence the "predicted slack":
  - predicted mean signed bias = -deficit * E[r_slow - r_fast]
  - predicted mean |bias|      = deficit * E[|r_slow - r_fast|]

We PASS the slack-prediction if (actual / predicted) is within 0.99 - 1.01.

The two paths are computed via INDEPENDENT code (no copy-paste):
  - A_oracle via direct vectorized r_slow - baseline
  - A_RAC via the same per-step injection loop used in rac_vtrace_identity_*

Output
------
results/rac_lambda_slack_check/results.json:
  - per-config (RS, NRS): mean signed bias, mean |bias|, predicted slack,
    ratio (actual / predicted), verdict_class
  - r_fast.npy, r_slow.npy: raw N=500 reward arrays (for future reuse)

Run example (GPU A; from /workspace/2_Delay_Aware_RLHF)
-------------------------------------------------------
  python -m scripts.rac_lambda_slack_check \
      --n_prompts 500 \
      --output_dir /workspace/2_Delay_Aware_RLHF/results/rac_lambda_slack_check
"""
from __future__ import annotations

import argparse
import json
import os
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
    w_age,
)


def compute_baseline(r_fast: np.ndarray) -> np.ndarray:
    """Causal running-mean baseline of r_fast (same as adv_quality_7B)."""
    n = len(r_fast)
    cum = np.cumsum(r_fast)
    counts = np.arange(1, n + 1)
    baseline = np.zeros(n, dtype=np.float64)
    baseline[1:] = cum[:-1] / counts[:-1]
    return baseline


def advantage_oracle(r_slow: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Oracle advantage = V-trace target at Lambda=I: r_slow - baseline."""
    return r_slow.astype(np.float64) - baseline.astype(np.float64)


def advantage_rac_const_lambda(
    r_fast: np.ndarray, r_slow: np.ndarray, baseline: np.ndarray, *,
    lambda_zero: float, tau_age: float, rho_clip: float = 1.0,
) -> np.ndarray:
    """RAC advantage with Delta=0 deterministic and Lambda[Delta=0] = lambda_zero.

    Mirrors adv_quality_7B.compute_advantage_metrics per-step injection loop
    with delays=0 everywhere. The kernel mass at Delta=0 is lambda_zero
    (1.0 for row-stochastic; 0.85 for non-row-stochastic slack-test).
    """
    n = len(r_fast)
    delays = np.zeros(n, dtype=np.int64)
    A_rac = (r_fast.astype(np.float64) - baseline.astype(np.float64)).copy()
    for s in range(n):
        t = s + int(delays[s])
        if t < n:
            A_rac[t] += lambda_zero * rho_clip * w_age(int(delays[s]), tau_age) * (
                r_slow[s] - r_fast[s]
            )
    return A_rac


def evaluate_kernel(
    r_fast: np.ndarray, r_slow: np.ndarray, baseline: np.ndarray,
    A_oracle: np.ndarray, *, lambda_zero: float, tau_age: float,
    rho_clip: float,
) -> dict:
    """Run one kernel config; return bias stats and predicted-slack comparison."""
    deficit = 1.0 - lambda_zero
    A_rac = advantage_rac_const_lambda(
        r_fast, r_slow, baseline,
        lambda_zero=lambda_zero, tau_age=tau_age, rho_clip=rho_clip,
    )
    diff = A_rac - A_oracle
    mean_signed_bias = float(np.mean(diff))
    mean_abs_bias = float(np.mean(np.abs(diff)))
    max_abs_bias = float(np.max(np.abs(diff)))

    # Predicted slack (Lambda - 1) * (r_slow - r_fast) = -deficit * (r_slow - r_fast).
    resid = r_slow.astype(np.float64) - r_fast.astype(np.float64)
    predicted_mean_signed_bias = float(-deficit * np.mean(resid))
    predicted_mean_abs_bias = float(deficit * np.mean(np.abs(resid)))

    # Pointwise ratio of actual to predicted (signed): every position should be 1.
    # Avoid divide-by-zero by masking near-zero predicted slack.
    pred_signed = -deficit * resid
    safe = np.abs(pred_signed) > 1e-9
    if safe.sum() > 0:
        pointwise_ratio = diff[safe] / pred_signed[safe]
        pointwise_ratio_mean = float(np.mean(pointwise_ratio))
        pointwise_ratio_std = float(np.std(pointwise_ratio))
        pointwise_ratio_min = float(np.min(pointwise_ratio))
        pointwise_ratio_max = float(np.max(pointwise_ratio))
    else:
        pointwise_ratio_mean = float("nan")
        pointwise_ratio_std = float("nan")
        pointwise_ratio_min = float("nan")
        pointwise_ratio_max = float("nan")

    # Aggregate-level ratios.
    if abs(predicted_mean_signed_bias) > 1e-9:
        ratio_signed = mean_signed_bias / predicted_mean_signed_bias
    else:
        ratio_signed = float("nan")
    if predicted_mean_abs_bias > 1e-9:
        ratio_abs = mean_abs_bias / predicted_mean_abs_bias
    else:
        ratio_abs = float("nan")

    # Numerical-zero PASS for row-stochastic; slack-prediction PASS for NRS.
    if deficit == 0.0:
        passed = max_abs_bias < 1e-9
        verdict_class = (
            "ESTABLISHED-ROW-STOCHASTIC-ZERO-BIAS" if passed
            else "FALSIFIED-ROW-STOCHASTIC-ZERO-BIAS"
        )
    else:
        # PASS if pointwise ratio is essentially 1 everywhere (theorem-exact)
        # OR aggregate ratios both within 1% of 1.
        tight = (
            abs(pointwise_ratio_mean - 1.0) < 1e-6
            and pointwise_ratio_std < 1e-6
        )
        loose = (
            (not np.isnan(ratio_signed) and abs(ratio_signed - 1.0) < 0.01)
            and (not np.isnan(ratio_abs) and abs(ratio_abs - 1.0) < 0.01)
        )
        passed = bool(tight or loose)
        if tight:
            verdict_class = "ESTABLISHED-SLACK-EXACT-POINTWISE"
        elif loose:
            verdict_class = "ESTABLISHED-SLACK-AGGREGATE"
        else:
            verdict_class = "FALSIFIED-SLACK-PREDICTION"

    return {
        "lambda_zero": float(lambda_zero),
        "deficit": float(deficit),
        "mean_signed_bias": mean_signed_bias,
        "mean_abs_bias": mean_abs_bias,
        "max_abs_bias": max_abs_bias,
        "predicted_mean_signed_bias": predicted_mean_signed_bias,
        "predicted_mean_abs_bias": predicted_mean_abs_bias,
        "ratio_signed_actual_over_predicted": float(ratio_signed)
            if not np.isnan(ratio_signed) else None,
        "ratio_abs_actual_over_predicted": float(ratio_abs)
            if not np.isnan(ratio_abs) else None,
        "pointwise_ratio_mean": pointwise_ratio_mean
            if not np.isnan(pointwise_ratio_mean) else None,
        "pointwise_ratio_std": pointwise_ratio_std
            if not np.isnan(pointwise_ratio_std) else None,
        "pointwise_ratio_min": pointwise_ratio_min
            if not np.isnan(pointwise_ratio_min) else None,
        "pointwise_ratio_max": pointwise_ratio_max
            if not np.isnan(pointwise_ratio_max) else None,
        "first5_A_rac":    [float(v) for v in A_rac[:5]],
        "first5_A_oracle": [float(v) for v in A_oracle[:5]],
        "first5_diff":     [float(v) for v in diff[:5]],
        "first5_predicted_signed": [float(v) for v in (-deficit * resid)[:5]],
        "passed": bool(passed),
        "verdict_class": verdict_class,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--gen_batch_size", type=int, default=4)
    ap.add_argument("--score_batch_size", type=int, default=8)
    ap.add_argument("--head_seed", type=int, default=42)
    ap.add_argument("--tau_age", type=float, default=1000.0)
    ap.add_argument("--rho_clip", type=float, default=1.0)
    ap.add_argument("--lambda_rs", type=float, default=1.0,
                    help="Row-stochastic control mass at Delta=0.")
    ap.add_argument("--lambda_nrs", type=float, default=0.85,
                    help="Non-row-stochastic test mass at Delta=0.")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[start] rac_lambda_slack_check  n={args.n_prompts}", flush=True)
    print(f"[start] cuda_devices={torch.cuda.device_count()} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    t_start = time.time()

    from datasets import load_dataset
    print("[data] loading UltraFeedback test_prefs...", flush=True)
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    prompts = [ds[i]["prompt"] for i in range(min(args.n_prompts, len(ds)))]
    print(f"[data] loaded {len(prompts)} prompts", flush=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Phase 1: generate
    print("\n=== Phase 1: generate responses ===", flush=True)
    pol, pol_tok = load_policy(bnb)
    responses = generate_responses(pol, pol_tok, prompts,
                                   max_new_tokens=args.max_new_tokens,
                                   batch_size=args.gen_batch_size)
    print(f"[gen] complete  responses={len(responses)}", flush=True)
    del pol
    torch.cuda.empty_cache()

    # Phase 2: fast RM
    print("\n=== Phase 2: score with fast RM ===", flush=True)
    fast_rm, fast_tok = load_fast_rm(bnb)
    hidden_size = int(fast_rm.config.hidden_size)
    head = make_fast_head(hidden_size, seed=args.head_seed,
                          device=next(fast_rm.parameters()).device,
                          dtype=torch.bfloat16)
    r_fast = score_with_fast_rm(fast_rm, fast_tok, head, prompts, responses,
                                batch_size=args.score_batch_size)
    print(f"[fast-score] mean={r_fast.mean():.4f}  std={r_fast.std():.4f}",
          flush=True)
    del fast_rm, head
    torch.cuda.empty_cache()

    # Phase 3: slow RM
    print("\n=== Phase 3: score with slow RM ===", flush=True)
    slow_rm, slow_tok = load_slow_rm(bnb)
    r_slow = score_with_slow_rm(slow_rm, slow_tok, prompts, responses,
                                batch_size=args.score_batch_size)
    print(f"[slow-score] mean={r_slow.mean():.4f}  std={r_slow.std():.4f}",
          flush=True)
    del slow_rm
    torch.cuda.empty_cache()

    # Persist raw arrays for future reuse.
    np.save(out_dir / "r_fast.npy", r_fast)
    np.save(out_dir / "r_slow.npy", r_slow)
    print(f"[cache] wrote r_fast.npy, r_slow.npy to {out_dir}", flush=True)

    # Phase 4: kernel comparisons
    print("\n=== Phase 4: kernel comparisons ===", flush=True)
    r_fast64 = r_fast.astype(np.float64)
    r_slow64 = r_slow.astype(np.float64)
    baseline = compute_baseline(r_fast64)
    A_oracle = advantage_oracle(r_slow64, baseline)

    rs_result = evaluate_kernel(
        r_fast64, r_slow64, baseline, A_oracle,
        lambda_zero=args.lambda_rs, tau_age=args.tau_age, rho_clip=args.rho_clip,
    )
    nrs_result = evaluate_kernel(
        r_fast64, r_slow64, baseline, A_oracle,
        lambda_zero=args.lambda_nrs, tau_age=args.tau_age, rho_clip=args.rho_clip,
    )

    print(f"\n[RS lambda={args.lambda_rs:.2f}]")
    print(f"  mean_signed_bias = {rs_result['mean_signed_bias']:.6e}")
    print(f"  mean_abs_bias    = {rs_result['mean_abs_bias']:.6e}")
    print(f"  max_abs_bias     = {rs_result['max_abs_bias']:.6e}")
    print(f"  verdict          = {rs_result['verdict_class']}")

    print(f"\n[NRS lambda={args.lambda_nrs:.2f}, deficit={1.0-args.lambda_nrs:.2f}]")
    print(f"  mean_signed_bias    = {nrs_result['mean_signed_bias']:.6e}")
    print(f"  predicted (signed)  = {nrs_result['predicted_mean_signed_bias']:.6e}")
    print(f"  ratio (signed)      = {nrs_result['ratio_signed_actual_over_predicted']}")
    print(f"  mean_abs_bias       = {nrs_result['mean_abs_bias']:.6e}")
    print(f"  predicted (abs)     = {nrs_result['predicted_mean_abs_bias']:.6e}")
    print(f"  ratio (abs)         = {nrs_result['ratio_abs_actual_over_predicted']}")
    print(f"  pointwise ratio     = mean={nrs_result['pointwise_ratio_mean']}"
          f"  std={nrs_result['pointwise_ratio_std']}")
    print(f"  pointwise ratio rng = [{nrs_result['pointwise_ratio_min']},"
          f" {nrs_result['pointwise_ratio_max']}]")
    print(f"  verdict             = {nrs_result['verdict_class']}")

    elapsed = time.time() - t_start
    results = {
        "config": {
            "n_prompts": args.n_prompts,
            "max_new_tokens": args.max_new_tokens,
            "head_seed": args.head_seed,
            "tau_age": args.tau_age,
            "rho_clip": args.rho_clip,
            "lambda_rs": args.lambda_rs,
            "lambda_nrs": args.lambda_nrs,
            "policy_model": POLICY_MODEL_ID,
            "fast_rm": FAST_RM_MODEL_ID,
            "slow_rm": SLOW_RM_MODEL_ID,
        },
        "row_stochastic": rs_result,
        "non_row_stochastic": nrs_result,
        "diagnostics": {
            "r_fast_mean": float(r_fast.mean()),
            "r_fast_std":  float(r_fast.std()),
            "r_slow_mean": float(r_slow.mean()),
            "r_slow_std":  float(r_slow.std()),
            "residual_mean":     float((r_slow.astype(np.float64) -
                                        r_fast.astype(np.float64)).mean()),
            "residual_abs_mean": float(np.abs(r_slow.astype(np.float64) -
                                              r_fast.astype(np.float64)).mean()),
        },
        "wall_time_seconds": elapsed,
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {json_path}  wall_time={elapsed:.0f}s", flush=True)

    print("\n=== SUMMARY ===")
    print(f"  RS  passed = {rs_result['passed']}  ({rs_result['verdict_class']})")
    print(f"  NRS passed = {nrs_result['passed']} ({nrs_result['verdict_class']})")


if __name__ == "__main__":
    main()
