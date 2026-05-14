"""V-trace clipped importance-sampling ratio rho_i^clip for RAC."""
from __future__ import annotations

import torch


def clipped_is_ratio(
    log_pi_current: torch.Tensor,
    log_pi_behavior: torch.Tensor,
    rho_bar: float = 1.0,
    *,
    epsilon: float | None = None,
) -> torch.Tensor:
    """V-trace canonical 1-sided clip: rho_i^clip = min(rho_bar, pi_current/pi_behavior).

    Aligns with paper Assumption A1 (rho_clip <= rho_bar a.s.) on which
    Proposition 1's TV bound depends. Computed in log-space for numerical
    stability, then upper-clipped to ``rho_bar``.

    Parameters
    ----------
    log_pi_current : torch.Tensor
        Log-probability under the current (target) policy pi_{theta(t+Delta)}.
    log_pi_behavior : torch.Tensor
        Log-probability under the behavior policy pi_{theta(t)}.
    rho_bar : float, default 1.0
        Upper clip threshold. V-trace canonical default = 1.0.
    epsilon : float, optional
        Deprecated. Retained as a keyword alias for backward compatibility
        with callers that previously passed the 2-sided PPO-style clip
        ``epsilon``; if supplied it is interpreted as ``rho_bar = 1.0 +
        epsilon`` so callers using ``epsilon=0.0`` recover ``rho_bar=1.0``.
        New code should pass ``rho_bar`` directly.
    """
    if epsilon is not None:
        # Map the legacy 2-sided clip-amount to the V-trace upper bound.
        rho_bar = 1.0 + float(epsilon)

    log_ratio = log_pi_current - log_pi_behavior
    ratio = torch.exp(log_ratio)
    return torch.minimum(torch.full_like(ratio, float(rho_bar)), ratio)
