"""CI guards for the Theorem A2+ composite (Pinsker + Bretagnolle-Huber) bound.

Pins the formulas in scripts/verify_a2_bretagnolle_huber.py so that:
  - Pinsker is monotone, zero at KL=0, matches √(½·KL).
  - BH is monotone, zero at KL=0, strictly below 1 for all finite KL, matches
    √(1 − e^{-KL}).
  - Composite min is ≤ each individual bound.
  - Crossover where BH < Pinsker is at KL ≈ 1.5936.
  - At KL=2.0, BH < Pinsker (Pinsker is borderline trivial).
  - At KL=3.0, Pinsker is trivial (≥1) while BH is informative (<1).
  - A short MC sanity check: 100% bound validity across a small N.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "scripts" / "verify_a2_bretagnolle_huber.py"

spec = importlib.util.spec_from_file_location("bh_mod", MODULE_PATH)
bh_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bh_mod)


# ---------------------------------------------------------------------------
# Formula identity
# ---------------------------------------------------------------------------

def test_pinsker_matches_formula():
    for kl in [0.0, 0.01, 0.5, 1.0, 2.0, 5.0, 10.0]:
        assert math.isclose(bh_mod.pinsker_bound(kl), math.sqrt(0.5 * kl),
                            rel_tol=1e-12, abs_tol=1e-12)


def test_bh_matches_formula():
    for kl in [0.0, 0.01, 0.5, 1.0, 2.0, 5.0, 10.0]:
        assert math.isclose(bh_mod.bh_bound(kl), math.sqrt(1.0 - math.exp(-kl)),
                            rel_tol=1e-12, abs_tol=1e-12)


def test_composite_matches_min():
    for kl in [0.0, 0.01, 0.5, 1.0, 1.594, 2.0, 3.0, 5.0]:
        assert math.isclose(
            bh_mod.composite_bound(kl),
            min(bh_mod.pinsker_bound(kl), bh_mod.bh_bound(kl)),
            rel_tol=1e-12, abs_tol=1e-12,
        )


# ---------------------------------------------------------------------------
# Boundary behavior
# ---------------------------------------------------------------------------

def test_bounds_zero_at_kl_zero():
    assert bh_mod.pinsker_bound(0.0) == 0.0
    assert bh_mod.bh_bound(0.0) == 0.0
    assert bh_mod.composite_bound(0.0) == 0.0


def test_bh_strictly_below_one():
    """BH = √(1-e^{-KL}) < 1 for all finite KL in the regime of interest.

    Note: double-precision underflow makes BH(KL) round to exactly 1.0 for
    KL ≳ 36, but in the RAC regime (KL ≤ ε, typically ε ≤ 3), BH stays
    strictly below 1.
    """
    for kl in [0.01, 0.5, 2.0, 5.0, 10.0, 20.0]:
        assert bh_mod.bh_bound(kl) < 1.0


def test_pinsker_trivial_at_kl_2():
    """Pinsker ≥ 1 at KL ≥ 2 — the bound degenerates (trivially true TV ≤ 1)."""
    for kl in [2.0, 3.0, 5.0, 10.0]:
        assert bh_mod.pinsker_bound(kl) >= 1.0


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

def test_pinsker_monotone():
    kl_grid = np.linspace(0, 10, 200)
    vals = [bh_mod.pinsker_bound(k) for k in kl_grid]
    diffs = np.diff(vals)
    assert (diffs >= -1e-12).all(), "Pinsker must be monotone non-decreasing"


def test_bh_monotone():
    kl_grid = np.linspace(0, 10, 200)
    vals = [bh_mod.bh_bound(k) for k in kl_grid]
    diffs = np.diff(vals)
    assert (diffs >= -1e-12).all(), "BH must be monotone non-decreasing"


def test_composite_monotone():
    kl_grid = np.linspace(0, 10, 200)
    vals = [bh_mod.composite_bound(k) for k in kl_grid]
    diffs = np.diff(vals)
    assert (diffs >= -1e-12).all(), "Composite must be monotone non-decreasing"


# ---------------------------------------------------------------------------
# Crossover & relative-tightness
# ---------------------------------------------------------------------------

def test_pinsker_tighter_below_crossover():
    """For KL < 1.5936 (the crossover), Pinsker should be the tighter bound."""
    for kl in [0.01, 0.1, 0.5, 1.0, 1.5]:
        assert bh_mod.pinsker_bound(kl) <= bh_mod.bh_bound(kl), (
            f"KL={kl}: Pinsker should be ≤ BH below crossover"
        )


def test_bh_tighter_above_crossover():
    """For KL > 1.5936, BH should be the tighter bound (strictly below Pinsker)."""
    for kl in [1.7, 2.0, 3.0, 5.0, 10.0]:
        assert bh_mod.bh_bound(kl) < bh_mod.pinsker_bound(kl), (
            f"KL={kl}: BH should be strictly < Pinsker above crossover"
        )


def test_crossover_is_at_1_5936():
    """Pinsker = BH at KL ≈ 1.5936 (solution of 1-e^{-KL} = KL/2)."""
    kl_star = 1.5936
    p = bh_mod.pinsker_bound(kl_star)
    b = bh_mod.bh_bound(kl_star)
    assert abs(p - b) < 1e-3, f"Bounds should nearly match at KL*: {p} vs {b}"


def test_bh_informative_while_pinsker_trivial_at_kl_3():
    """At KL=3, Pinsker ≥ 1 (trivial) but BH < 1 — composite ≡ BH, informative."""
    assert bh_mod.pinsker_bound(3.0) >= 1.0
    assert bh_mod.bh_bound(3.0) < 1.0
    assert math.isclose(bh_mod.composite_bound(3.0), bh_mod.bh_bound(3.0))


# ---------------------------------------------------------------------------
# Full advantage-bias prefactor
# ---------------------------------------------------------------------------

def test_advantage_bias_bound_prefactor():
    """2·V_max/(1-γ) prefactor applied correctly to each TV bound."""
    V_max = 10.0
    gamma = 0.9
    kl = 0.5
    prefactor = 2.0 * V_max / (1.0 - gamma)  # 200

    p = bh_mod.advantage_bias_bound(kl, V_max, gamma, bh_mod.pinsker_bound)
    b = bh_mod.advantage_bias_bound(kl, V_max, gamma, bh_mod.bh_bound)
    c = bh_mod.advantage_bias_bound(kl, V_max, gamma, bh_mod.composite_bound)

    assert math.isclose(p, prefactor * math.sqrt(0.5 * kl))
    assert math.isclose(b, prefactor * math.sqrt(1.0 - math.exp(-kl)))
    assert c == min(p, b)


# ---------------------------------------------------------------------------
# Short MC sanity check (small N for CI speed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_kl", [0.01, 0.5, 2.0])
def test_mc_bound_never_violated(target_kl):
    """A quick 300-trial MC at each regime: composite bound must hold
    100% of the time (0 violations). This catches regressions where someone
    edits one of the bound formulas by mistake.
    """
    rng = np.random.default_rng(42 + int(target_kl * 100))
    res = bh_mod.mc_trials_at_kl(target_kl, n_trials=300, rng=rng)
    assert res["composite_valid_frac"] == 1.0, (
        f"Composite bound violated in MC at target KL={target_kl}: "
        f"{res['composite_valid_frac']*100:.1f}% valid"
    )
    assert res["pinsker_valid_frac"] == 1.0
    assert res["bh_valid_frac"] == 1.0


def test_mc_tightness_sane_order_of_magnitude():
    """Empirical tightness ratio should be in the range (0.001, 1) — this is
    a sanity check that the MC harness isn't producing degenerate numbers."""
    rng = np.random.default_rng(0)
    res = bh_mod.mc_trials_at_kl(0.5, n_trials=500, rng=rng)
    assert 0.0 < res["tightness_ratio_pinsker"] < 1.0
    assert 0.0 < res["tightness_ratio_composite"] < 1.0


# ---------------------------------------------------------------------------
# JSON artifact smoke (exercises the full script if it has been run)
# ---------------------------------------------------------------------------

def test_json_artifact_has_expected_shape_if_present():
    """If the validation script has been run, its JSON must contain the
    key structural fields. Skip if not yet run (dev environments)."""
    import json
    path = REPO / "results" / "a2_bh" / "bh_validation.json"
    if not path.exists():
        pytest.skip("Validation JSON not yet written — run the script first")
    data = json.loads(path.read_text())
    assert "formulas" in data
    assert "regimes" in data
    assert "verdict" in data
    for regime in ("small_kl", "moderate_kl", "large_kl", "very_large_kl"):
        assert regime in data["regimes"]
        r = data["regimes"][regime]
        assert r["composite_valid_frac"] == 1.0
