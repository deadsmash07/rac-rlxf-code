# Retroactive Advantage Correction (RAC)

![RAC: a reward that arrives Δ steps late still belongs to the rollout that earned it. RAC queues each pending slow reward, ages it through a decay kernel, and reinjects it as a clipped residual into the optimizer step where it lands.](assets/rac_poster.png)

Code for **"Retroactive Advantage Correction: Closed-Form V-Trace Bias
Correction for Delay-Aware RLHF"**, accepted at the **ICML 2026 Workshop on
Reinforcement Learning from World Feedback (RLxF)**.

**▶ [Interactive explainer, with diagrams](https://deadsmash07.github.io/rac-rlxf-code/)** — a visual walkthrough of where a reward signal gets stuck Δ optimizer steps behind the gradient that should consume it, and how RAC sends it forward. (Also viewable locally: open `docs/index.html`.)

RAC is a forward-injection primitive for delay-aware
RLHF: each slow reward that arrives late is queued, aged through a non-negative
kernel, and reinjected as an additive correction into the next optimiser
step's advantage.

The repository covers three layers:

1. **`src/rac/`** is the core RAC implementation: the additive correction
   primitive, the rollout queue, the age kernel, the importance-sampling clip,
   and the tensor-shaped multi-channel delay kernel.
2. **`scripts/`** contains the reproduction scripts for every quantitative
   claim in the paper, including the closed-form `47.9x` headline at `K=2`,
   the cross-topology K-sweep, the heavy-tailed delay-distribution stress, the
   MDP-size scaling sweep, the lambda-slack-sweep at 7B, the V-trace
   identity-kernel collapse at 7B, and the static-batch advantage-quality
   probe at 7B.
3. **`figures/`** regenerates every figure in the paper from the JSON results.

The PPO/GRPO integration is implemented as a two-line reward-manager patch:
see `src/trainer/multi_channel_reward_manager.py` and `scripts/main_ppo_rac.py`.

## Repository layout

```
rac-rlxf-code/
  src/
    rac/                # core RAC primitive
      primitive.py
      advantage_corrector.py
      rollout_cache.py
      tensor_lambda.py
      importance_sampling.py
      age_discount.py
    trainer/            # PPO/GRPO reward-manager integration
    channels/           # mock + real reward channels used in 7B probes
  scripts/              # paper-claim reproductions (see below)
  figures/              # paper figure generation
  tests/                # unit + integration tests
  requirements.txt
```

## Requirements

The closed-form tabular benchmarks run on a single CPU thread in NumPy. The 7B
probes need a single H100 (4-bit NF4 inference for two reward heads of the
Qwen-2.5-7B + Skywork-Reward-Llama-3.1-8B class). The PPO/GRPO integration
extends the VERL/HybridFlow reward-manager interface.

```
pip install -r requirements.txt
```

Listed dependencies are pinned where reproducibility requires it.

## Experimental setup at a glance

| | Tabular MDP benchmark | 7B/8B reward-distribution probe |
|---|---|---|
| **Purpose** | closed-form policy bias vs. a known optimum | check RAC's core algebra on real reward signals |
| **Delay Δ** | optimizer steps: {1,…,5} (headline), {5,20,50}, up to {100,200} | per-step samples — deterministic Δ=5, lognormal, Pareto |
| **Rollout** | 3-state × 2-action MDP, 1000 trials/seed | N=500 UltraFeedback prompts, ≤128 response tokens, no PPO loop |
| **Reward channels** | synthetic: fast = truth + 𝒩(0, σ_f²), σ_f=0.5; slow = truth, delayed | fast = Qwen2.5-7B head; slow oracle = Skywork-Reward-Llama-3.1-8B; policy = Llama-3-8B |
| **How Δ is simulated** | FIFO buffer: a score computed at step *t* pops out and is reinjected at *t+Δ* | sample a delay per step, then forward-inject the residual RAC would add at *t+Δ* |

Δ is always measured in **optimizer (gradient) steps** — the number of steps a
slow reward is in flight before it returns, not wall-clock. The age kernel is
`w_age(Δ) = exp(−Δ/τ_age)`. The 7B probe is a static algebraic check (identity
actor `ρ_clip = 1`), not a training-speedup measurement; end-to-end LLM-scale
PPO training is stated as future work.

## Reproducing each paper claim

Each command writes its outputs under `results/` in the current working
directory.

### Table 1, top block: K-sweep at sigma_f=0.5, Delta_k in {1,...,5}

```
python scripts/ablate_rac_components_K2_47_9.py
```

Reproduces the K=2 closed-form `47.9x` headline (and the knob ablation for
Table 3 in Appendix C). 50 MDP seeds, 1000 trials per seed, bootstrap 95%
confidence intervals with `B=1000` resamples. Runs in roughly 30 seconds on a
single CPU thread.

### Table 1, bottom block: K=2 baseline comparison (wait-for-slow, Retrace-A, RAC)

```
python scripts/ablate_rac_components_K2_47_9.py --baselines
```

The same script with the `--baselines` flag emits the cost-quality Pareto
points for naive PPO, wait-for-slow, Retrace-A, and RAC at the `K=2`
deployment operating point `E[Delta]=25`.

### Appendix C: cross-topology K-sweep across five tabular topologies

```
python scripts/run_K_sweep_parallel_resume.py
python scripts/verify_K_sweep_cross_mdp_topology.py
python scripts/aggregate_K_sweep.py
```

Sweeps `K in {1,2,3,4,5,7,10,15,20}` across five MDP topologies (canonical
`3x2`, chain `5x2`, cyclic `4x3`, dense `5x3`, terminal `3x2`). The aggregate
script produces the JSON used by `figures/fig4_k_sweep_heatmap.py`.

### Appendix D: heavy-tailed delay-distribution stress

```
python scripts/verify_rac_heavy_tail_delay.py
```

Five distributions matched at `E[Delta]=20`: deterministic, clipped Gaussian,
lognormal, Pareto-finite, truncated-Cauchy. The truncated-Cauchy realised
mean is `~21.0` under the `[1,200]` clip and the paper discloses this offset.

### Appendix G: MDP-size scaling

```
python scripts/adv_mdp_scaling.py
```

Sweeps `(|S|, |A|) in {(3,2), (5,3), (10,5), (20,8)}` with five MDP-structure
seeds per size. Outputs the reduction range that the paper summarises as
`4.65 to 17.9x` order-of-magnitude reduction across MDPs in
`{3, 5, 10, 20}` states.

### Appendix B: lambda-slack sweep at 7B (linear-slack verification)

```
python scripts/rac_lambda_slack_check.py
python scripts/rac_lambda_slack_sweep.py
```

Sweeps the slack-deficit `eta in {0.05, 0.10, 0.15, 0.20, 0.30, 0.50}` on 500
UltraFeedback prompts scored by Qwen-2.5-7B (fast) and
Skywork-Reward-Llama-3.1-8B (slow). Verifies the `(1-eta)` form of Theorem 2.1
to machine precision (pointwise ratio `1.000000` with std `< 1e-15` per
slack value).

### Appendix G: V-trace identity-kernel collapse at 7B

```
python scripts/rac_vtrace_identity_kernel_check.py
```

Verifies `||A_RAC - A_V-trace||_inf = 0.0` in float-64 at `Lambda = I`,
`rho_clip = 1`, `w_age(0) = 1`. Both code paths use identical operation order
so the algebraic identity reproduces bit-exactly.

### Seed-replication of identity-kernel and slack-sweep

```
python scripts/rac_seed_replication.py
```

Replicates the identity-kernel collapse and the slack-sweep across Qwen
fast-head seeds `{42, 43, 44, 45}`. All four pass at machine precision.

### Appendix G: static-batch advantage-quality probe at 7B

```
python scripts/adv_quality_7B.py
```

Computes the `l_2` error and cosine similarity against the synchronous-oracle
advantage for the delayed-fast control and the RAC-corrected advantage, on
500 UltraFeedback prompts. The static-batch result is a single-seed
random-init probe at Pearson `r = -0.53` against the oracle. End-to-end PPO
validation across multiple seeds is left as future work.

### PPO/GRPO integration entry point

```
python scripts/main_ppo_rac.py --config configs/...
```

The reward-manager interface in `src/trainer/multi_channel_reward_manager.py`
adds `O(K)` queue maintenance and one tensor addition per optimiser step.
Wall-clock overhead on a 1.5B Qwen-2.5 PPO run sat within Monte-Carlo noise
of vanilla GRPO. The full end-to-end LLM-scale PPO run referenced in the
paper as future work was not in scope for this submission.

## Figures

Every figure in the paper regenerates from the JSON results produced by the
scripts above:

```
python figures/fig1_rac_schematic.py
python figures/fig2_delay_distributions.py
python figures/fig3_k2_bias_reduction_bar.py
python figures/fig4_k_sweep_heatmap.py
python figures/fig5_cost_quality_pareto.py
python figures/mdp_scaling.py
python figures/regenerate_adv_quality_7B_plot.py
```

`_figstyle.py` defines the Wong colorblind-safe palette and the matplotlib
publication-style helpers shared across figures.

## Tests

```
pytest tests/
```

Unit tests cover the RAC primitive (`test_rac_primitive.py`), the
advantage corrector (`test_advantage_corrector.py`), the rollout cache
(`test_rollout_cache.py`), forward-injection (`test_forward_injection.py`),
the delayed-affine ground-truth check (`test_rac_delayed_affine.py`), and
the gradient-validation invariants (`test_rac_gradient_validation.py`). The
KL-bias test (`test_kl_bias.py`) covers Proposition 2.2 numerically.
`test_a2_bretagnolle_huber.py` checks the Pinsker / Bretagnolle-Huber
crossover root `KL* = 1.625873915824`.

The integration test `test_rac_pipeline_integration.py` runs an end-to-end
smoke test through the reward-manager patch.

## Notes

- The closed-form K=2 tabular MDP benchmark reproduces in about 30 lines of
  NumPy on a single CPU thread.
- The heavy-tail stress test (`verify_rac_heavy_tail_delay.py`) runs in
  about 945 seconds on a single CPU thread for `2.25e5` trajectories.
- The 7B identity-kernel and slack-sweep verifications use 4-bit NF4
  inference with bf16 compute on a single H100.
- The `Lambda = I` boundary case in Theorem 2.1 recovers V-trace's
  on-policy guarantee from Espeholt et al. (2018).
- The PPO integration is implemented against the VERL/HybridFlow reward
  manager interface; the same two-line patch should apply to any
  reward-manager that exposes the standard PPO/GRPO call signature.

## Citation

```bibtex
@inproceedings{raj2026rac,
  title     = {Retroactive Advantage Correction: Closed-Form {V-Trace}
               Bias Correction for Delay-Aware {RLHF}},
  author    = {Raj, Arnav},
  booktitle = {ICML 2026 Workshop on Reinforcement Learning from
               World Feedback (RLxF)},
  year      = {2026}
}
```

## License

CC BY 4.0. See `LICENSE`.
