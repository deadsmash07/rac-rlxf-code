"""Lambda-slack SWEEP: extend single-point slack verification to a deficit grid.

Authoring metadata
------------------
- Skill citation: professional-rl-reviewer Section theorem-tightness +
  remote H100 launch dispatch.

What this script does
---------------------
Strengthens the single-point slack-verification from  (which
showed that at deficit eta=0.15 the empirical Lambda-slack matches the
predicted slack pointwise with ratio = 1.0 exactly) to a SWEEP across
eta in {0.05, 0.10, 0.15, 0.20, 0.30, 0.50}. If the ratio = 1.0 at every
eta, the theorem is tight as a LINEAR function of (1 - sum Lambda), not
just at a single point.

The cached r_fast.npy and r_slow.npy from the parent slack-check directory
are reused -- no LLM inference is needed for the sweep, which is pure
numpy. This makes the sweep CPU-bound and seconds-fast.

Output
------
results/rac_lambda_slack_sweep/results.json: per-eta empirical/predicted
mean signed and abs bias, pointwise ratio mean and std, verdict_class.
results/rac_lambda_slack_sweep/lambda_slack_sweep.pdf: 1-panel plot,
x=eta, y=mean abs bias (empirical and predicted overlaid).

Run example
-------------------------------------------------------
  python -m scripts.rac_lambda_slack_sweep \
      --cache_dir .//results/rac_lambda_slack_check \
      --output_dir .//results/rac_lambda_slack_sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rac_lambda_slack_check import (  # noqa: E402
    advantage_oracle,
    advantage_rac_const_lambda,
    compute_baseline,
    evaluate_kernel,
)


DEFAULT_ETAS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True,
                    help="Directory holding cached r_fast.npy, r_slow.npy.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tau_age", type=float, default=1000.0)
    ap.add_argument("--rho_clip", type=float, default=1.0)
    ap.add_argument("--etas", type=float, nargs="+", default=list(DEFAULT_ETAS))
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[start] rac_lambda_slack_sweep  etas={args.etas}", flush=True)
    t_start = time.time()

    r_fast = np.load(cache_dir / "r_fast.npy").astype(np.float64)
    r_slow = np.load(cache_dir / "r_slow.npy").astype(np.float64)
    n = len(r_fast)
    print(f"[load] r_fast.npy r_slow.npy  n={n}", flush=True)
    print(f"[load] r_fast mean={r_fast.mean():.4f} std={r_fast.std():.4f}",
          flush=True)
    print(f"[load] r_slow mean={r_slow.mean():.4f} std={r_slow.std():.4f}",
          flush=True)

    baseline = compute_baseline(r_fast)
    A_oracle = advantage_oracle(r_slow, baseline)

    per_eta = []
    print(f"\n=== sweep over {len(args.etas)} deficit values ===", flush=True)
    for eta in args.etas:
        lambda_zero = 1.0 - float(eta)
        res = evaluate_kernel(
            r_fast, r_slow, baseline, A_oracle,
            lambda_zero=lambda_zero, tau_age=args.tau_age,
            rho_clip=args.rho_clip,
        )
        res["eta"] = float(eta)
        per_eta.append(res)
        print(f"\n[eta={eta:.2f}  lambda_zero={lambda_zero:.2f}]")
        print(f"  empirical mean_signed_bias = "
              f"{res['mean_signed_bias']:.6e}")
        print(f"  predicted mean_signed_bias = "
              f"{res['predicted_mean_signed_bias']:.6e}")
        print(f"  empirical mean_abs_bias    = "
              f"{res['mean_abs_bias']:.6e}")
        print(f"  predicted mean_abs_bias    = "
              f"{res['predicted_mean_abs_bias']:.6e}")
        print(f"  ratio_signed               = "
              f"{res['ratio_signed_actual_over_predicted']}")
        print(f"  ratio_abs                  = "
              f"{res['ratio_abs_actual_over_predicted']}")
        print(f"  pointwise ratio mean       = "
              f"{res['pointwise_ratio_mean']}")
        print(f"  pointwise ratio std        = "
              f"{res['pointwise_ratio_std']}")
        print(f"  verdict                    = {res['verdict_class']}")

    elapsed = time.time() - t_start
    results = {
        "config": {
            "n_prompts": int(n),
            "tau_age": float(args.tau_age),
            "rho_clip": float(args.rho_clip),
            "etas": [float(e) for e in args.etas],
            "cache_dir": str(cache_dir),
            "reused_r_fast_npy": True,
            "reused_r_slow_npy": True,
        },
        "per_eta": per_eta,
        "diagnostics": {
            "r_fast_mean": float(r_fast.mean()),
            "r_fast_std":  float(r_fast.std()),
            "r_slow_mean": float(r_slow.mean()),
            "r_slow_std":  float(r_slow.std()),
            "residual_mean":     float((r_slow - r_fast).mean()),
            "residual_abs_mean": float(np.abs(r_slow - r_fast).mean()),
        },
        "wall_time_seconds": elapsed,
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {json_path}  wall_time={elapsed:.2f}s", flush=True)

    # ---- Figure: 1-panel, empirical vs predicted mean |bias| vs eta ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Wong colorblind-safe palette (inlined; no external dependency).
        WONG_BLUE = "#0072B2"
        WONG_VERMILLION = "#D55E00"
        WONG_GREY = "#606060"

        plt.rcParams.update({
            "font.family":       "serif",
            "font.serif":        ["Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset":  "stix",
            "font.size":         9,
            "axes.labelsize":    10,
            "axes.titlesize":    10,
            "axes.titleweight":  "bold",
            "axes.linewidth":    0.6,
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "axes.edgecolor":    WONG_GREY,
            "xtick.labelsize":   9,
            "ytick.labelsize":   9,
            "xtick.color":       WONG_GREY,
            "ytick.color":       WONG_GREY,
            "legend.fontsize":   8,
            "legend.frameon":    False,
            "pdf.fonttype":      42,
            "ps.fonttype":       42,
        })

        etas = np.array([r["eta"] for r in per_eta])
        emp = np.array([r["mean_abs_bias"] for r in per_eta])
        pred = np.array([r["predicted_mean_abs_bias"] for r in per_eta])

        fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=300)
        ax.plot(etas, pred, "-", color=WONG_BLUE, lw=1.4,
                label=r"Predicted $\eta\,\mathbb{E}[|r^{\mathrm{slow}}-r^{\mathrm{fast}}|]$",
                zorder=2)
        ax.plot(etas, emp, "o", color=WONG_VERMILLION, ms=5.5,
                mfc="none", mew=1.2,
                label="Empirical mean $|\\mathrm{bias}|$", zorder=3)
        ax.set_xlabel(r"Deficit $\eta = 1 - \sum_\Delta \Lambda$")
        ax.set_ylabel(r"Mean $|\mathrm{bias}|$ (advantage units)")
        ax.set_xlim(0.0, max(etas) * 1.05)
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="upper left", handletextpad=0.5)
        ax.grid(True, axis="y", lw=0.3, color=WONG_GREY, alpha=0.4)
        fig.tight_layout(pad=0.4)
        fig_path = out_dir / "lambda_slack_sweep.pdf"
        fig.savefig(fig_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] wrote {fig_path}", flush=True)
    except Exception as e:
        print(f"[fig] FAILED to render: {e!r}", flush=True)

    # Summary line
    all_tight = all(
        (r.get("pointwise_ratio_mean") is not None
         and abs(r["pointwise_ratio_mean"] - 1.0) < 1e-6
         and r["pointwise_ratio_std"] is not None
         and r["pointwise_ratio_std"] < 1e-14)
        for r in per_eta
    )
    print("\n=== SWEEP SUMMARY ===")
    for r in per_eta:
        print(f"  eta={r['eta']:.2f}  ratio_pointwise_mean="
              f"{r['pointwise_ratio_mean']}  std={r['pointwise_ratio_std']}"
              f"  verdict={r['verdict_class']}")
    print(f"  ALL_LINEAR_EXACT = {all_tight}")


if __name__ == "__main__":
    main()
