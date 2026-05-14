"""Tensor-Λ gradient estimator (Theorem A3).

Generalizes GRPO-λ's scalar λ to rank-2 tensor Λ[k, Δ] over reward channels k
and wall-clock delays Δ.

ĝ_Λ = Σ_k Σ_Δ Λ[k, Δ] · δ_{k, Δ} · ∇_θ log π_{θ(t+Δ)}(o_i | q_i)

Unbiased (MC-validated 2026-04-16, EXACT under (U-i) across 3 KL drift
scenarios) iff:
  (U-i)   Σ_Δ Λ[k, Δ] = 1 for every channel k
  (U-ii)  Λ[k, Δ] = 0 for Δ > D_max (deterministic horizon)
  (U-iii) IS correction ρ_i^clip applied per (k, Δ) entry

Corollaries:
  GRPO-λ = Λ[k=1, Δ=0]
  Retrace = Λ[k=1, Δ=trajectory-time]
  V-trace = similar, different clip
  RAC is NOT a Λ slice (Corollary A3.4) — w_age lives outside (U-i); treat
  tensor-Λ and RAC's age-discount as separate contributions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch


@dataclass
class TensorLambda:
    """Rank-2 tensor Λ[k, Δ] indexed by channel k and delay Δ.

    Attributes
    ----------
    weights : dict[int, dict[int, float]]
        weights[k][Δ] = Λ[k, Δ]. Entries not present default to 0.
    n_channels : int
    D_max : int
        Maximum delay horizon.
    """

    weights: dict[int, dict[int, float]] = field(default_factory=dict)
    n_channels: int = 1
    D_max: int = 10

    def set(self, k: int, delta: int, value: float) -> None:
        if delta > self.D_max:
            raise ValueError(f"delta {delta} > D_max {self.D_max}")
        self.weights.setdefault(k, {})[delta] = value

    def get(self, k: int, delta: int) -> float:
        return self.weights.get(k, {}).get(delta, 0.0)

    def per_channel_sum(self, k: int) -> float:
        return sum(self.weights.get(k, {}).values())

    def validate(self, tol: float = 1e-8) -> None:
        """Assert (U-i) Σ_Δ Λ[k,Δ] = 1 per channel, (U-ii) truncated."""
        for k in range(self.n_channels):
            s = self.per_channel_sum(k)
            if abs(s - 1.0) > tol:
                raise ValueError(
                    f"(U-i) violated for channel k={k}: sum(Λ[k,Δ])={s} ≠ 1"
                )
            for delta in self.weights.get(k, {}):
                if delta > self.D_max:
                    raise ValueError(f"(U-ii) violated: Λ[{k},{delta}]≠0 but {delta}>D_max")


def validate_uniformity(Lambda: TensorLambda) -> None:
    """Alias for Lambda.validate() — accepts and raises on violation."""
    Lambda.validate()


def tensor_lambda_gradient(
    Lambda: TensorLambda,
    delta_kD: dict[tuple[int, int], torch.Tensor],
    rho_clip: dict[tuple[int, int], torch.Tensor],
    grad_log_pi: dict[tuple[int, int], torch.Tensor],
) -> torch.Tensor:
    """Compute ĝ_Λ = Σ_k Σ_Δ Λ[k,Δ] · δ_{k,Δ} · ∇log π.

    Parameters
    ----------
    Lambda : TensorLambda
    delta_kD : dict keyed by (k, Δ) → tensor of δ values per example
    rho_clip : dict keyed by (k, Δ) → tensor of IS ratios
    grad_log_pi : dict keyed by (k, Δ) → tensor of ∇log π evaluated at θ(t+Δ)

    Returns
    -------
    g : torch.Tensor, the unbiased-under-(U-i) policy gradient estimate
    """
    Lambda.validate()
    g_total = None
    for (k, delta), coef in [((k, d), Lambda.get(k, d)) for k in range(Lambda.n_channels)
                              for d in range(Lambda.D_max + 1)
                              if Lambda.get(k, d) != 0.0]:
        key = (k, delta)
        term = coef * rho_clip[key] * delta_kD[key] * grad_log_pi[key]
        g_total = term if g_total is None else g_total + term
    assert g_total is not None, "Tensor-Λ is entirely zero"
    return g_total
