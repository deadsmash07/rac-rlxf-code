"""Static-batch advantage-quality demonstration for RAC at 7B real-LLM scale.

Authoring metadata
------------------
- Dispatch target: RunPod GPU A (H100 80GB; pod port 17321)
- Skill citation: professional-rl-reviewer §static-quality-metric +
  launch-runpod-h100-job §dispatch.

What this script does
---------------------
NO PPO training. Static-batch fixed-policy advantage-quality probe demonstrating
that RAC's corrected advantage estimate is closer to the synchronous oracle than
the uncorrected delayed-fast control. Steps:

1. Load Llama-3-8B-Instruct, generate N=500 greedy responses to UltraFeedback
   test_prefs prompts (one response per prompt, max_new_tokens=128).
2. Score each (prompt, response) pair with both reward models:
   - r_fast: Qwen2.5-7B-Instruct backbone + ad-hoc Linear(hidden, 1) head with
     fixed seed=42 (mirrors the TwoChannelRewardModule fast-RM construction
     from prior fast-RM construction for reproducibility; head is RANDOM-init, frozen).
   - r_slow: Skywork-Reward-Llama-3.1-8B-v0.2 native scalar head (the "oracle").
3. For each delay channel (deterministic Δ=5, lognormal μ=1.5 σ=0.8, pareto α=2.5):
     - sample one delay per step (n=500)
     - compute three advantage signals over t=0..499:
         baseline_t = running mean of r_fast up to t
         A_sync[t]    = r_slow[t] - baseline_t                  (oracle advantage)
         A_control[t] = r_fast[t] - baseline_t                  (delayed-only fast)
         A_rac[t]     = r_fast[t] - baseline_t
                        + Σ_{s: s + delays[s] == t} w_age(delays[s]) *
                                                     (r_slow[s] - r_fast[s])
       IS ratio = 1 (frozen policy => identity actor; see RAC §method default ρ=1).
       w_age uses τ_age=1000 → exp(-Δ/1000) ≈ 0.99..1.00 on the operating Δ grid
       (matches paper's geometric-kernel constant-on-operating-range regime).
4. Metrics per channel:
     err_control = ‖A_control - A_sync‖_2
     err_rac     = ‖A_rac - A_sync‖_2
     bias_reduction = err_control / err_rac          (>1 means RAC closer to oracle)
     cos_control = cos(A_control, A_sync)
     cos_rac     = cos(A_rac, A_sync)
5. Save JSON + matplotlib plot (Wong palette, 2-panel: bias_reduction bar +
   cosine bar).

This is the static-quality-metric equivalent of the K=2 closed-form ratio
headline lifted from synthetic tabular MDP to 7B real-LLM rewards over a real
text dataset (UltraFeedback). It does NOT require PPO training — fixed-policy
fixed-trajectory advantage estimator quality only.

Implementation invariants
-------------------------
- 4-bit nf4 quantization for both RMs (fit two 7B+8B models in 80 GB H100).
- bf16 compute dtype.
- batch_size=4 for generation; batch_size=8 for scoring.
- Right-padded sequences using each tokenizer's pad_id.
- Llama-3 backbone for generation: NousResearch/Meta-Llama-3-8B-Instruct
  (chat-template via tokenizer.apply_chat_template).
- All RNG seeded: rng_seed=42 for delay sampling and head init.
- No PPO; no LoRA; no gradient flow.

Run example (GPU A; from /workspace/2_Delay_Aware_RLHF)
-------------------------------------------------------
  python -m scripts.adv_quality_7B \
      --n_prompts 500 \
      --output_dir /workspace/2_Delay_Aware_RLHF/results/adv_quality_7B
"""
from __future__ import annotations

import argparse
import json
import math
import os
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


# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------
POLICY_MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
FAST_RM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SLOW_RM_MODEL_ID = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"


# ---------------------------------------------------------------------------
# Delay channels (paper §heavy-tail and headline regime)
# ---------------------------------------------------------------------------
def sample_delays(channel: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample one delay per step.  Returns int array shape (n,) with delays >= 1.

    deterministic: Δ=5 constant (matches the §3 K=2 working-point Δ-grid).
    lognormal:     μ=1.5 σ=0.8 (paper Appendix D heavy-tail probe).
    pareto:        α=2.5 (paper Appendix D heavy-tail probe).
    """
    if channel == "deterministic":
        return np.full(n, 5, dtype=np.int64)
    if channel == "lognormal":
        d = rng.lognormal(mean=1.5, sigma=0.8, size=n)
        return np.clip(np.round(d), 1, None).astype(np.int64)
    if channel == "pareto":
        # numpy's pareto gives (x_min=1) Lomax; +1 to get classic Pareto with
        # support [1, inf). Then floor to discrete delay.
        d = (rng.pareto(2.5, size=n) + 1.0) * 2.0  # scale to mean ~ matching others
        return np.clip(np.round(d), 1, None).astype(np.int64)
    raise ValueError(f"Unknown channel: {channel}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def load_policy(bnb: BitsAndBytesConfig):
    print(f"[load] policy = {POLICY_MODEL_ID}", flush=True)
    tok = AutoTokenizer.from_pretrained(POLICY_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left pad for batched generate
    model = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL_ID, torch_dtype=torch.bfloat16, quantization_config=bnb,
        device_map={"": 0},
    )
    model.generation_config.pad_token_id = tok.pad_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.requires_grad_(False)
    model.train(False)
    return model, tok


def generate_responses(model, tok, prompts: list[str], *, max_new_tokens: int,
                       batch_size: int) -> list[str]:
    responses: list[str] = []
    device = next(model.parameters()).device
    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        # strip prompt
        gen = out[:, enc["input_ids"].shape[1] :]
        decoded = tok.batch_decode(gen, skip_special_tokens=True)
        responses.extend(decoded)
        if (i // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (len(prompts) - i - batch_size) / max(i + batch_size, 1)
            print(f"[gen] {i + batch_size}/{len(prompts)}  elapsed={elapsed:.0f}s "
                  f"eta={eta:.0f}s", flush=True)
    return responses


# ---------------------------------------------------------------------------
# Reward scoring
# ---------------------------------------------------------------------------
def load_fast_rm(bnb: BitsAndBytesConfig):
    print(f"[load] fast RM = {FAST_RM_MODEL_ID}", flush=True)
    tok = AutoTokenizer.from_pretrained(FAST_RM_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        FAST_RM_MODEL_ID, torch_dtype=torch.bfloat16, quantization_config=bnb,
        device_map={"": 0},
    )
    model.requires_grad_(False)
    model.train(False)
    return model, tok


def load_slow_rm(bnb: BitsAndBytesConfig):
    print(f"[load] slow RM = {SLOW_RM_MODEL_ID}", flush=True)
    tok = AutoTokenizer.from_pretrained(SLOW_RM_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        SLOW_RM_MODEL_ID, num_labels=1, torch_dtype=torch.bfloat16,
        quantization_config=bnb, device_map={"": 0}, trust_remote_code=True,
    )
    model.config.pad_token_id = tok.pad_token_id
    model.requires_grad_(False)
    model.train(False)
    return model, tok


def make_fast_head(hidden_size: int, *, seed: int, device, dtype) -> nn.Linear:
    """Random-init Linear(hidden, 1) scoring head, seeded for reproducibility.
    Mirrors TwoChannelRewardModule's score-head init pattern (uniform[-bound, bound],
    bound = sqrt(6/hidden), no bias).
    """
    head = nn.Linear(hidden_size, 1, bias=False)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    bound = math.sqrt(6.0 / hidden_size)
    with torch.no_grad():
        w = (torch.rand(hidden_size, generator=gen) * 2 - 1) * bound
        head.weight.copy_(w.view(1, -1))
    return head.to(device=device, dtype=dtype)


def score_with_fast_rm(model, tok, head, prompts: list[str], responses: list[str],
                       *, batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    scores: list[float] = []
    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        bp = prompts[i : i + batch_size]
        br = responses[i : i + batch_size]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p},
                 {"role": "assistant", "content": r}],
                tokenize=False, add_generation_prompt=False,
            )
            for p, r in zip(bp, br)
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states[-1]  # (B, L, H)
            attn = enc["attention_mask"]  # (B, L)
            # last non-pad token index per row (right-pad)
            seq_lens = attn.sum(dim=1) - 1  # (B,)
            idx = seq_lens.view(-1, 1, 1).expand(-1, 1, hs.size(-1))
            last_h = hs.gather(1, idx).squeeze(1)  # (B, H)
            r = head(last_h).squeeze(-1).float().detach().cpu().numpy()
        scores.extend(r.tolist())
        if (i // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[fast-score] {i + batch_size}/{len(prompts)}  elapsed={elapsed:.0f}s",
                  flush=True)
    return np.asarray(scores, dtype=np.float64)


def score_with_slow_rm(model, tok, prompts: list[str], responses: list[str],
                       *, batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    scores: list[float] = []
    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        bp = prompts[i : i + batch_size]
        br = responses[i : i + batch_size]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p},
                 {"role": "assistant", "content": r}],
                tokenize=False, add_generation_prompt=False,
            )
            for p, r in zip(bp, br)
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(device)
        with torch.no_grad():
            out = model(**enc, return_dict=True)
            r = out.logits.squeeze(-1).float().detach().cpu().numpy()
        scores.extend(r.tolist())
        if (i // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[slow-score] {i + batch_size}/{len(prompts)}  elapsed={elapsed:.0f}s",
                  flush=True)
    return np.asarray(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# Advantage construction and metrics
# ---------------------------------------------------------------------------
def w_age(delta: int, tau_age: float = 1000.0) -> float:
    return float(math.exp(-float(delta) / tau_age))


def compute_advantage_metrics(r_fast: np.ndarray, r_slow: np.ndarray,
                              delays: np.ndarray,
                              *, tau_age: float = 1000.0) -> dict:
    """Compute the three advantage sequences and L2 + cosine metrics."""
    n = len(r_fast)
    # running-mean baseline of r_fast (causal; baseline at t uses r_fast[0..t-1])
    cum = np.cumsum(r_fast)
    counts = np.arange(1, n + 1)
    # baseline at t = mean of r_fast[0..t-1]; at t=0 use 0
    baseline = np.zeros(n)
    baseline[1:] = cum[:-1] / counts[:-1]
    # Synchronous oracle advantage: r_slow returned synchronously at t
    A_sync = r_slow - baseline
    # Delayed-only control: r_fast at t (slow signal discarded)
    A_control = r_fast - baseline
    # RAC: r_fast at t plus forward-injected residual from any rollout s where
    # s + delays[s] == t. The injection lands at t (forward by Δ_s steps).
    A_rac = A_control.copy()
    for s in range(n):
        t = s + int(delays[s])
        if t < n:
            A_rac[t] += w_age(int(delays[s]), tau_age) * (r_slow[s] - r_fast[s])
    # Metrics
    err_control = float(np.linalg.norm(A_control - A_sync))
    err_rac = float(np.linalg.norm(A_rac - A_sync))
    bias_reduction = err_control / max(err_rac, 1e-12)
    def cos(a, b):
        na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    cos_control = cos(A_control, A_sync)
    cos_rac = cos(A_rac, A_sync)
    return {
        "err_control": err_control,
        "err_rac": err_rac,
        "bias_reduction": bias_reduction,
        "cos_control": cos_control,
        "cos_rac": cos_rac,
        "mean_delay": float(np.mean(delays)),
        "max_delay": int(np.max(delays)),
        "n_injected": int(np.sum((np.arange(n) + delays) < n)),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(per_channel: dict, output_path: Path) -> None:
    """Wong-palette 2-panel: panel A bias_reduction bar, panel B cosine bar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    WONG = {
        "orange":     "#E69F00",
        "sky_blue":   "#56B4E9",
        "blu_green":  "#009E73",
        "vermillion": "#D55E00",
        "grey":       "#606060",
        "blue":       "#0072B2",
    }
    plt.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size":        8,
        "axes.labelsize":   9,
        "axes.titlesize":   10,
        "axes.titleweight": "bold",
        "axes.linewidth":   0.6,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.edgecolor":   WONG["grey"],
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "xtick.color":      WONG["grey"],
        "ytick.color":      WONG["grey"],
        "legend.fontsize":  7.5,
        "legend.frameon":   False,
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    channels = list(per_channel.keys())
    chan_labels = {"deterministic": "Det $\\Delta{=}5$",
                   "lognormal": "Lognormal $\\mu{=}1.5,\\sigma{=}0.8$",
                   "pareto": "Pareto $\\alpha{=}2.5$"}
    labels = [chan_labels.get(c, c) for c in channels]
    bias_red = [per_channel[c]["bias_reduction"] for c in channels]
    cos_ctrl = [per_channel[c]["cos_control"] for c in channels]
    cos_rac  = [per_channel[c]["cos_rac"] for c in channels]
    x = np.arange(len(channels))

    # Panel A: bias reduction (single bar per channel)
    ax = axes[0]
    bars = ax.bar(x, bias_red, color=WONG["blue"], width=0.55, edgecolor="black",
                  linewidth=0.4)
    for xi, v in zip(x, bias_red):
        ax.text(xi, v * 1.02, f"{v:.2f}$\\times$",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color=WONG["grey"], linestyle="--", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Bias-reduction ratio (err$_{\\mathrm{control}}$/err$_{\\mathrm{RAC}}$)")
    ax.set_title("A. RAC advantage L2-error reduction (oracle target)")
    ax.set_ylim(bottom=0)
    ax.text(0.98, 0.95, f"$\\tau_{{age}}{{=}}1000$, $N{{=}}500$",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color=WONG["grey"])

    # Panel B: cosine sim, control vs RAC (paired bars)
    ax = axes[1]
    width = 0.36
    ax.bar(x - width/2, cos_ctrl, width=width, color=WONG["vermillion"],
           edgecolor="black", linewidth=0.4, label="Control (delayed-fast)")
    ax.bar(x + width/2, cos_rac,  width=width, color=WONG["blu_green"],
           edgecolor="black", linewidth=0.4, label="RAC")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Cosine similarity to A$_{\\mathrm{sync}}$")
    ax.set_title("B. Cosine alignment with synchronous oracle")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=1)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[plot] wrote {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--gen_batch_size", type=int, default=4)
    ap.add_argument("--score_batch_size", type=int, default=8)
    ap.add_argument("--rng_seed", type=int, default=42)
    ap.add_argument("--head_seed", type=int, default=42)
    ap.add_argument("--tau_age", type=float, default=1000.0)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[start] adv_quality_7B  n={args.n_prompts}  seed={args.rng_seed}",
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
    print(f"[gen] complete  responses={len(responses)}  "
          f"first50_chars[0]={responses[0][:50]!r}", flush=True)
    # Free generation policy from GPU memory before loading the RMs
    del pol
    torch.cuda.empty_cache()
    print(f"[mem] cuda_alloc={torch.cuda.memory_allocated() / 1e9:.2f} GB "
          f"reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB", flush=True)

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
    print(f"[fast-score] complete  mean={r_fast.mean():.4f}  std={r_fast.std():.4f}  "
          f"min={r_fast.min():.4f}  max={r_fast.max():.4f}", flush=True)
    del fast_rm, head
    torch.cuda.empty_cache()
    print(f"[mem] cuda_alloc={torch.cuda.memory_allocated() / 1e9:.2f} GB "
          f"reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB", flush=True)

    # --- Phase 3: slow RM (oracle) scoring -----------------------------------
    print("\n=== Phase 3: score with slow RM (Skywork) ===", flush=True)
    slow_rm, slow_tok = load_slow_rm(bnb)
    r_slow = score_with_slow_rm(slow_rm, slow_tok, prompts, responses,
                                batch_size=args.score_batch_size)
    print(f"[slow-score] complete  mean={r_slow.mean():.4f}  std={r_slow.std():.4f}  "
          f"min={r_slow.min():.4f}  max={r_slow.max():.4f}", flush=True)
    del slow_rm
    torch.cuda.empty_cache()

    # Cross-RM diagnostic
    fmean = float(r_fast.mean()); smean = float(r_slow.mean())
    fcen = r_fast - fmean; scen = r_slow - smean
    denom = math.sqrt(float((fcen ** 2).sum()) * float((scen ** 2).sum()))
    pearson = float((fcen * scen).sum() / max(denom, 1e-12))
    print(f"[diag] pearson(r_fast, r_slow) = {pearson:.4f}  "
          f"|r_fast - r_slow|.mean = {float(np.abs(r_fast - r_slow).mean()):.4f}",
          flush=True)

    # --- Phase 4: per-channel advantage metrics ------------------------------
    print("\n=== Phase 4: per-channel advantage metrics ===", flush=True)
    rng = np.random.default_rng(args.rng_seed)
    per_channel: dict[str, dict] = {}
    for channel in ["deterministic", "lognormal", "pareto"]:
        delays = sample_delays(channel, len(r_fast), rng)
        m = compute_advantage_metrics(r_fast, r_slow, delays, tau_age=args.tau_age)
        per_channel[channel] = m
        print(f"[adv] {channel:13s}  mean_delay={m['mean_delay']:.2f}  "
              f"max_delay={m['max_delay']}  injected={m['n_injected']}  "
              f"err_ctrl={m['err_control']:.3f}  err_rac={m['err_rac']:.3f}  "
              f"bias_red={m['bias_reduction']:.2f}x  "
              f"cos_ctrl={m['cos_control']:.3f}  cos_rac={m['cos_rac']:.3f}",
              flush=True)

    # --- Persist --------------------------------------------------------------
    elapsed = time.time() - t_start
    results = {
        "config": {
            "n_prompts": args.n_prompts,
            "max_new_tokens": args.max_new_tokens,
            "rng_seed": args.rng_seed,
            "head_seed": args.head_seed,
            "tau_age": args.tau_age,
            "policy_model": POLICY_MODEL_ID,
            "fast_rm": FAST_RM_MODEL_ID,
            "slow_rm": SLOW_RM_MODEL_ID,
        },
        "diagnostics": {
            "r_fast_mean": fmean, "r_fast_std": float(r_fast.std()),
            "r_slow_mean": smean, "r_slow_std": float(r_slow.std()),
            "pearson_fast_slow": pearson,
        },
        "per_channel": per_channel,
        "wall_time_seconds": elapsed,
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {json_path}  wall_time={elapsed:.0f}s", flush=True)

    # Plot
    plot_path = out_dir / "adv_quality_7B.pdf"
    make_plot(per_channel, plot_path)

    # Final summary print
    print("\n=== SUMMARY ===")
    for ch, m in per_channel.items():
        print(f"  {ch:13s}  bias_reduction={m['bias_reduction']:.3f}x  "
              f"cos_control={m['cos_control']:.3f}  cos_rac={m['cos_rac']:.3f}")
    print(f"max_bias_reduction = "
          f"{max(m['bias_reduction'] for m in per_channel.values()):.3f}x")


if __name__ == "__main__":
    main()
