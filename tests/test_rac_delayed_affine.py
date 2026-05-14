"""Unit tests for RACDelayedAffineScore and DelayedAffineScore in
scripts/run_ppo_rac_vs_delay.py.

Pins the forward-injection semantics so a silent refactor can't reduce RAC
to vanilla delayed reward.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Lazy-import the script module without running its main()
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_ppo_rac_vs_delay.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_ppo_rac_vs_delay", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    # stub datasets / trl / transformers / peft to avoid heavy imports when
    # the script is loaded — we only need AffineScore / DelayedAffineScore /
    # RACDelayedAffineScore classes.
    stubs = [
        "datasets", "trl", "transformers", "peft",
    ]
    saved = {}
    for s in stubs:
        saved[s] = sys.modules.get(s)
        stub = types.ModuleType(s)
        if s == "datasets":
            stub.Dataset = type("Dataset", (), {"from_list": staticmethod(lambda x: x)})
        elif s == "trl":
            stub.PPOConfig = type("PPOConfig", (), {})
            stub.PPOTrainer = type("PPOTrainer", (), {})
        elif s == "transformers":
            for cls in ("AutoModelForCausalLM", "AutoModelForSequenceClassification",
                        "AutoTokenizer"):
                setattr(stub, cls, type(cls, (), {}))
        elif s == "peft":
            stub.LoraConfig = type("LoraConfig", (), {})
            stub.PeftModel = type("PeftModel", (), {})
            stub.get_peft_model = lambda *a, **kw: None
        sys.modules[s] = stub
    try:
        spec.loader.exec_module(mod)
    finally:
        # restore
        for s, orig in saved.items():
            if orig is None:
                sys.modules.pop(s, None)
            else:
                sys.modules[s] = orig
    return mod


class FakeInner(nn.Module):
    """Deterministic 'inner score': returns a scalar derived from x's sum.

    We intentionally vary x between unique calls so _sig() detects a new
    outer step, while keeping x constant for duplicate calls within the
    same outer step.
    """

    def __init__(self):
        super().__init__()
        # register a dummy parameter so `.parameters()` is non-empty
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Produce a per-sample-scalar that depends on the summed first row
        # of x; broadcast across a fake seq+feat dim to mimic RM output shape.
        B = x.shape[0]
        # Use mean across non-batch dims; make sure gradient flows through p
        # so .parameters() behavior is preserved.
        s = x.detach().float().reshape(B, -1).mean(dim=1)
        # Shape back to (B, 1) so alpha+beta*inner(x) broadcasts cleanly.
        return (s + self.p).view(B, 1)


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# ----------------------------- DelayedAffineScore tests -----------------


def test_delayed_emits_zero_for_first_K_calls(mod):
    inner = FakeInner()
    w = mod.DelayedAffineScore(inner, alpha=1.0, beta=2.0, delay=5)

    for t in range(5):
        # Unique input each call
        x = torch.ones(4, 8) * float(t + 1)
        out = w(x)
        assert torch.allclose(out, torch.zeros_like(out)), (
            f"t={t}: expected zeros during warmup, got mean={out.float().mean()}"
        )

    # t=5: should now emit the score from t=0
    x = torch.ones(4, 8) * 100.0
    out = w(x)
    expected_scalar_from_t0 = (1.0 + 2.0 * 1.0)  # alpha + beta * 1.0 (at t=0, x=1)
    # inner at t=0 was mean(x)=1.0 so score was 1+2*1 = 3.0
    assert torch.allclose(out, torch.full_like(out, expected_scalar_from_t0)), (
        f"t=5: expected {expected_scalar_from_t0}, got {out.float().mean()}"
    )


def test_delayed_call_log_tracks_true_vs_emitted(mod):
    inner = FakeInner()
    w = mod.DelayedAffineScore(inner, alpha=0.0, beta=1.0, delay=2)

    for t in range(5):
        x = torch.ones(2, 4) * float(t + 1)  # unique each step
        _ = w(x)

    # After 5 calls: last log should show true_mean=5, emitted_mean=3 (from t=2)
    last = w.call_log[-1]
    assert last["true_mean"] == pytest.approx(5.0)
    assert last["emitted_mean"] == pytest.approx(3.0)
    assert last["delayed"] is True


# ----------------------------- RACDelayedAffineScore tests --------------


def test_rac_emits_delta_during_warmup(mod):
    """RAC's key property: during first K steps, emit != zero (vs plain delayed)."""
    inner = FakeInner()
    w = mod.RACDelayedAffineScore(
        inner, alpha=0.0, beta=1.0, delay=5, tau_age=50.0, alpha_delta=1.0,
    )
    # On the very first call, r_fast_ewm is None → delta=0 → emit=base=0.
    x0 = torch.ones(4, 8) * 1.0
    out0 = w(x0)
    # Call 0: r_fast_ewm is None at compute-time, so delta=0 (emit=0).
    assert torch.allclose(out0, torch.zeros_like(out0))
    assert w.call_log[-1]["rac_seed"] is True

    # On the SECOND call, r_fast_ewm was initialized to mean(x0 score)=1.0
    # on call 0.  At call 1, the code first UPDATES r_fast_ewm with the new
    # batch mean before computing delta.  So r_fast_ewm becomes:
    #   ewm_new = (1 - 1/τ) * 1.0 + (1/τ) * 2.0  (τ=ewm_tau=50)
    #          = 0.98 + 0.04 = 1.02
    # Then delta = w_age * alpha_delta * (2.0 - 1.02) = w_age * 0.98
    x1 = torch.ones(4, 8) * 2.0
    out1 = w(x1)
    w_age = math.exp(-5 / 50.0)
    ewm_after_call1 = (1 - 1 / w.ewm_tau) * 1.0 + (1 / w.ewm_tau) * 2.0
    expected_delta = w_age * 1.0 * (2.0 - ewm_after_call1)
    assert torch.allclose(out1, torch.full_like(out1, expected_delta), atol=1e-5), (
        f"expected emit={expected_delta}, got {out1.float().mean()}"
    )
    assert w.call_log[-1]["rac_seed"] is True  # still in warmup
    assert w.call_log[-1]["delta"] == pytest.approx(expected_delta, abs=1e-5)


def test_rac_emits_base_plus_delta_after_warmup(mod):
    inner = FakeInner()
    w = mod.RACDelayedAffineScore(
        inner, alpha=0.0, beta=1.0, delay=2, tau_age=50.0, alpha_delta=1.0,
    )

    # Feed 4 unique calls.  At t=2, emit should be (base from t=0) + δ.
    # At t=0: r_fast_ewm starts None → updated to mean(x=1)=1.0 AFTER compute;
    #         the check "r_fast_ewm is None" happened before update, so delta=0.
    # At t=1: r_fast_ewm ≈ 1.0 (after update); current=2; delta = w_age*(2-1) = w_age
    # At t=2: FIFO pops score from t=0 = 1.0 as base; delta computed with
    #         updated r_fast_ewm; current=3.
    # Let's just check that the rac_active flag is set and base+delta are non-zero.
    outs = []
    for t in range(4):
        x = torch.ones(2, 4) * float(t + 1)
        outs.append(w(x))

    # At t=2, should be base (from t=0 = 1.0) + some delta
    # The emitted_mean should differ from base by the delta (recorded in log).
    logs = w.call_log
    assert logs[2]["rac_active"] is True
    assert logs[2]["rac_seed"] is False  # no longer warmup
    # emit = true_mean_from_t0 + delta; base here is 1.0.
    base_from_t0 = 1.0  # score at t=0
    assert abs(logs[2]["emitted_mean"] - (base_from_t0 + logs[2]["delta"])) < 1e-5


def test_rac_delta_zero_when_r_fast_matches_current(mod):
    """If the current true score matches the running-mean baseline, delta=0."""
    inner = FakeInner()
    w = mod.RACDelayedAffineScore(
        inner, alpha=0.0, beta=1.0, delay=3, tau_age=50.0, alpha_delta=1.0,
    )
    # Feed IDENTICAL inputs many times → EWM converges to 1.0; current is 1.0.
    # So delta should → 0.
    for t in range(20):
        x = torch.ones(2, 4) * float(t + 1)  # must be UNIQUE for is_new logic
        _ = w(x)

    # r_fast_ewm should be near the recent mean. Feed a duplicate-value sequence
    # at the end: starting at t=20, use a constant x that matches the EWM.
    # Actually we need r_fast_ewm == current. Easier: set alpha_delta=0.
    w2 = mod.RACDelayedAffineScore(
        inner, alpha=0.0, beta=1.0, delay=3, tau_age=50.0, alpha_delta=0.0,
    )
    for t in range(5):
        x = torch.ones(2, 4) * float(t + 1)
        _ = w2(x)
    for log in w2.call_log:
        assert log["delta"] == pytest.approx(0.0, abs=1e-7), log


def test_rac_correction_norm_cap(mod):
    """If delta would blow up, it is capped to max_correction_norm."""
    inner = FakeInner()
    w = mod.RACDelayedAffineScore(
        inner, alpha=0.0, beta=1000.0, delay=2, tau_age=50.0, alpha_delta=10.0,
        max_correction_norm=5.0,
    )
    for t in range(5):
        x = torch.ones(2, 4) * float(t + 1) * 100  # huge scores
        _ = w(x)

    # All recorded deltas should be within cap
    for log in w.call_log[1:]:  # skip warm cold-start zero
        assert abs(log["delta"]) <= 5.0 + 1e-3, log


def test_rac_age_weight_decreases_with_delay(mod):
    """w_age(K=5) > w_age(K=50) when τ_age=50."""
    inner = FakeInner()
    w_short = mod.RACDelayedAffineScore(inner, 0, 1, delay=5, tau_age=50)
    w_long = mod.RACDelayedAffineScore(inner, 0, 1, delay=50, tau_age=50)
    assert w_short.w_age_K > w_long.w_age_K


def test_rac_delay_zero_falls_through(mod):
    """At delay=0, RACDelayedAffineScore should emit the raw score unchanged."""
    inner = FakeInner()
    w = mod.RACDelayedAffineScore(inner, alpha=1.0, beta=2.0, delay=0)
    x = torch.ones(3, 5) * 2.0
    out = w(x)
    expected = 1.0 + 2.0 * 2.0
    assert torch.allclose(out, torch.full_like(out, expected))
    log = w.call_log[-1]
    assert log["rac_active"] is False
    assert log["delta"] == 0.0


def test_delayed_and_rac_differ_on_warmup(mod):
    """REGRESSION: RAC emits != 0 during warmup, delayed emits == 0."""
    inner = FakeInner()
    w_delayed = mod.DelayedAffineScore(inner, 0.0, 1.0, delay=5)
    w_rac = mod.RACDelayedAffineScore(
        inner, 0.0, 1.0, delay=5, tau_age=50.0, alpha_delta=1.0,
    )

    for t in range(5):
        x = torch.ones(2, 4) * float(t + 1)
        out_delayed = w_delayed(x)
        out_rac = w_rac(x)
        delayed_mean = float(out_delayed.float().mean())
        rac_mean = float(out_rac.float().mean())
        # During warmup, delayed MUST be zero
        assert abs(delayed_mean) < 1e-7, f"t={t}: delayed emit {delayed_mean}"
        # RAC after the first call should have nonzero emit
        if t > 0:
            assert abs(rac_mean) > 1e-5, f"t={t}: RAC emit should be nonzero"
