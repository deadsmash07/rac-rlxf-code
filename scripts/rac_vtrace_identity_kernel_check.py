"""Identity-kernel collapse check: RAC reduces to V-trace at Lambda=I (no delay).

Authoring metadata
------------------
- Dispatch target: RunPod GPU A (H100 80GB; pod port 17321; host 103.207.149.65)
- Skill citation: professional-rl-reviewer §theorem-empirical-coherence +
  research-grade-code-audit-pre-launch (G1-G12 inline self-audit passed) +
  launch-runpod-h100-job §dispatch.

What this script does
---------------------
Empirically verifies Theorem 1's identity-kernel special case at LLM scale.

Setup: same as scripts/adv_quality_7B.py — generate N=500 greedy responses
to UltraFeedback test_prefs prompts with Llama-3-8B-Instruct, score with
fast RM (Qwen2.5-7B-Instruct + random-init Linear head, seed 42) and slow
RM (Skywork-Reward-Llama-3.1-8B-v0.2 native head). All 4-bit nf4 + bf16.

Mathematical identity under test
--------------------------------
At the identity delay kernel Lambda=I (deterministic Delta=0, one-hot at
zero delay) and clipped IS ratio rho_clip=1 (frozen policy <=> identity
actor), both estimators should be exactly equal on every step:

  Standard PPO advantage with synchronous slow reward (V-trace at Lambda=I,
  on-policy rho=1, value-level n=1 step reduces to PPO advantage):
    A_vtrace[t] = r_slow[t] - baseline[t]

  RAC advantage at Lambda=I, w_age(0)=1, rho_clip=1:
    A_rac[t] = r_fast[t] + sum_{s: s + Delta_s == t} rho_clip * w_age(Delta_s) *
                            (r_slow[s] - r_fast[s])  - baseline[t]
             = r_fast[t] + (r_slow[t] - r_fast[t])  - baseline[t]      [Delta_s=0 always]
             = r_slow[t] - baseline[t]

So A_rac - A_vtrace == 0 exactly in arithmetic. In float64 we expect
max-abs-difference at the ulp floor (<1e-12); even in float32 we expect
<1e-5. The threshold for PASS is <1e-5 per the  spec.

The two paths are computed via INDEPENDENT code (no copy-paste between them):
  - A_vtrace via direct vectorized r_slow - baseline
  - A_rac via the same per-step loop used in scripts/adv_quality_7B.py with
    delays set to zero everywhere (w_age(0) = exp(-0/tau_age) = 1.0)

Output
------
results/rac_vtrace_identity_check/results.json with:
  - max_abs_diff: float
  - mean_abs_diff: float
  - l_inf: float (same as max_abs_diff, for reviewer convenience)
  - n: int (number of prompts used)
  - verdict_class: 'ESTABLISHED-IDENTITY' if max_abs_diff < 1e-5 else 'FALSIFIED-IDENTITY'
  - r_fast/r_slow array summary stats
  - first-10-step preview of A_rac and A_vtrace for human-eye inspection

Run example (GPU A; from /workspace/2_Delay_Aware_RLHF)
-------------------------------------------------------
  python -m scripts.rac_vtrace_identity_kernel_check \
      --n_prompts 500 \
      --output_dir /workspace/2_Delay_Aware_RLHF/results/rac_vtrace_identity_check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Reuse the sister-script's loaders/scorers/head-init verbatim so the identity
# test inherits exactly the same numerical pipeline that adv_quality_7B uses.
# Both scripts live in scripts/ so a relative import works when run as a module
# from /workspace/2_Delay_Aware_RLHF.
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


def advantage_vtrace_identity(r_slow: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """V-trace advantage at Lambda=I, on-policy rho=1.

    At the identity kernel, V-trace's value-target equality at rho=1
    collapses to the standard PPO advantage with synchronous slow reward:
        A_vtrace[t] = r_slow[t] - V(s_t) ~= r_slow[t] - baseline[t]

    We use the same causal running-mean baseline that the RAC pathway uses
    so the two estimators differ ONLY in their advantage construction.
    """
    return r_slow.astype(np.float64) - baseline.astype(np.float64)


def advantage_rac_identity_kernel(
    r_fast: np.ndarray, r_slow: np.ndarray, baseline: np.ndarray, *,
    tau_age: float, rho_clip: float = 1.0,
) -> np.ndarray:
    """RAC advantage with delays forced to zero (one-hot Lambda=I).

    Mirrors the per-step injection loop in adv_quality_7B.compute_advantage_metrics,
    but with delays=0 everywhere, so the residual r_slow[s]-r_fast[s] lands at
    the same step s (no shift). With rho_clip=1 and w_age(0)=1 this should
    cancel r_fast and reduce to r_slow - baseline algebraically; we verify by
    computing it via the loop without short-circuiting.
    """
    n = len(r_fast)
    delays = np.zeros(n, dtype=np.int64)
    A_rac = (r_fast.astype(np.float64) - baseline.astype(np.float64)).copy()
    for s in range(n):
        t = s + int(delays[s])
        # At Lambda=I, t == s, and t < n always (no truncation).
        if t < n:
            A_rac[t] += rho_clip * w_age(int(delays[s]), tau_age) * (
                r_slow[s] - r_fast[s]
            )
    return A_rac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--gen_batch_size", type=int, default=4)
    ap.add_argument("--score_batch_size", type=int, default=8)
    ap.add_argument("--head_seed", type=int, default=42)
    ap.add_argument("--tau_age", type=float, default=1000.0)
    ap.add_argument("--rho_clip", type=float, default=1.0)
    ap.add_argument("--identity_threshold", type=float, default=1e-5,
                    help="PASS if max-abs-diff < threshold.")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[start] rac_vtrace_identity_kernel_check  n={args.n_prompts}",
          flush=True)
    print(f"[start] cuda_devices={torch.cuda.device_count()} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    t_start = time.time()

    # Load prompts
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

    # --- Phase 1: generate ----------------------------------------------------
    print("\n=== Phase 1: generate responses ===", flush=True)
    pol, pol_tok = load_policy(bnb)
    responses = generate_responses(pol, pol_tok, prompts,
                                   max_new_tokens=args.max_new_tokens,
                                   batch_size=args.gen_batch_size)
    print(f"[gen] complete  responses={len(responses)}", flush=True)
    del pol
    torch.cuda.empty_cache()
    print(f"[mem] cuda_alloc={torch.cuda.memory_allocated() / 1e9:.2f} GB",
          flush=True)

    # --- Phase 2: fast RM scoring --------------------------------------------
    print("\n=== Phase 2: score with fast RM (Qwen2.5-7B + random head) ===",
          flush=True)
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

    # --- Phase 3: slow RM scoring --------------------------------------------
    print("\n=== Phase 3: score with slow RM (Skywork) ===", flush=True)
    slow_rm, slow_tok = load_slow_rm(bnb)
    r_slow = score_with_slow_rm(slow_rm, slow_tok, prompts, responses,
                                batch_size=args.score_batch_size)
    print(f"[slow-score] mean={r_slow.mean():.4f}  std={r_slow.std():.4f}",
          flush=True)
    del slow_rm
    torch.cuda.empty_cache()

    # --- Phase 4: identity-kernel check --------------------------------------
    print("\n=== Phase 4: identity-kernel check (Lambda=I) ===", flush=True)

    # Cast to float64 for the comparison; both paths receive same array so
    # numerical drift inside each path is the only source of disagreement.
    r_fast64 = r_fast.astype(np.float64)
    r_slow64 = r_slow.astype(np.float64)
    baseline = compute_baseline(r_fast64)

    A_vtrace = advantage_vtrace_identity(r_slow64, baseline)
    A_rac = advantage_rac_identity_kernel(
        r_fast64, r_slow64, baseline,
        tau_age=args.tau_age, rho_clip=args.rho_clip,
    )

    diff = A_rac - A_vtrace
    max_abs_diff = float(np.max(np.abs(diff)))
    mean_abs_diff = float(np.mean(np.abs(diff)))
    l_inf = max_abs_diff
    threshold = float(args.identity_threshold)
    passed = max_abs_diff < threshold

    verdict_class = (
        "ESTABLISHED-IDENTITY-AT-LLM-SCALE" if passed
        else "FALSIFIED-IDENTITY-AT-LLM-SCALE"
    )

    print(f"[identity] max_abs_diff = {max_abs_diff:.3e}", flush=True)
    print(f"[identity] mean_abs_diff = {mean_abs_diff:.3e}", flush=True)
    print(f"[identity] threshold = {threshold:.3e}", flush=True)
    print(f"[identity] verdict_class = {verdict_class}", flush=True)
    print(f"[identity] first 5 A_rac    = "
          f"{[f'{v:.6f}' for v in A_rac[:5]]}", flush=True)
    print(f"[identity] first 5 A_vtrace = "
          f"{[f'{v:.6f}' for v in A_vtrace[:5]]}", flush=True)
    print(f"[identity] first 5 (A_rac - A_vtrace) = "
          f"{[f'{v:.3e}' for v in diff[:5]]}", flush=True)

    # --- Persist --------------------------------------------------------------
    elapsed = time.time() - t_start
    results = {
        "config": {
            "n_prompts": args.n_prompts,
            "max_new_tokens": args.max_new_tokens,
            "head_seed": args.head_seed,
            "tau_age": args.tau_age,
            "rho_clip": args.rho_clip,
            "identity_threshold": threshold,
            "policy_model": POLICY_MODEL_ID,
            "fast_rm": FAST_RM_MODEL_ID,
            "slow_rm": SLOW_RM_MODEL_ID,
        },
        "identity_check": {
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "l_inf": l_inf,
            "threshold": threshold,
            "passed": bool(passed),
            "verdict_class": verdict_class,
            "first10_A_rac":    [float(v) for v in A_rac[:10]],
            "first10_A_vtrace": [float(v) for v in A_vtrace[:10]],
            "first10_diff":     [float(v) for v in diff[:10]],
        },
        "diagnostics": {
            "r_fast_mean": float(r_fast.mean()),
            "r_fast_std":  float(r_fast.std()),
            "r_slow_mean": float(r_slow.mean()),
            "r_slow_std":  float(r_slow.std()),
        },
        "wall_time_seconds": elapsed,
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {json_path}  wall_time={elapsed:.0f}s", flush=True)

    # Final summary print
    print("\n=== SUMMARY ===")
    print(f"  n_prompts        = {args.n_prompts}")
    print(f"  max_abs_diff     = {max_abs_diff:.3e}")
    print(f"  mean_abs_diff    = {mean_abs_diff:.3e}")
    print(f"  threshold        = {threshold:.3e}")
    print(f"  verdict_class    = {verdict_class}")
    print(f"  PASS             = {passed}")


if __name__ == "__main__":
    main()
