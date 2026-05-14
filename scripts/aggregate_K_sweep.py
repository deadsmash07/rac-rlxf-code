"""Aggregate K-channel sweep (K ∈ {3, 5, 7, 10}) into one table ().

Reads the per-K `validation.json` outputs from
`results/track2_tensor_lambda_multichannel{,_K5,_K7,_K10}/` and emits:

- `results/track2_tensor_lambda_multichannel/K_sweep_summary.json`
- a markdown table on stdout (copy-pasteable into paper §4.2)

Paper claim being quantified: "RAC bias stays near the Monte-Carlo floor as
channel count K grows, while naive fast-only bias grows with K (because the
fast channel carries only 1/K of the true signal)." If the reduction-factor
column grows monotonically with K at matched MC budget, Theorem A3's
per-channel unbiasedness claim is empirically tight.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT_DIR = RES / "track2_tensor_lambda_multichannel"


def load_one(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    m = d["metrics"]
    cfg = d["config"]
    return {
        "K": cfg["n_channels"],
        "n_trials": cfg["n_trials"],
        "seeds": len(cfg["seeds"]),
        "bias_fast": m["bias_fast"],
        "bias_tL": m["bias_tL"],
        "var_fast": m["var_fast"],
        "var_tL": m["var_tL"],
        "reduction_factor": m["reduction_factor_fast"],
        "vif_vs_fast": m["vif_vs_fast"],
        "pass": d.get("overall_pass", True),
    }


def main():
    rows = []
    for subdir in [
        "track2_tensor_lambda_multichannel",          # K=3
        "track2_tensor_lambda_multichannel_K5",
        "track2_tensor_lambda_multichannel_K7",
        "track2_tensor_lambda_multichannel_K10",
    ]:
        r = load_one(RES / subdir / "validation.json")
        if r is not None:
            rows.append(r)
    rows.sort(key=lambda x: x["K"])

    out_path = OUT_DIR / "K_sweep_summary.json"
    out_path.write_text(json.dumps(rows, indent=2))

    # markdown table
    print("| K | n_trials | bias_fast | bias_RAC | reduction× | VIF | pass |")
    print("|---|---------:|----------:|---------:|-----------:|----:|:----:|")
    for r in rows:
        print(f"| {r['K']} | {r['n_trials']} | {r['bias_fast']:.4f} | {r['bias_tL']:.5f} | "
              f"**{r['reduction_factor']:.1f}×** | {r['vif_vs_fast']:.2f} | "
              f"{'✓' if r['pass'] else '✗'} |")
    print(f"\n[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
