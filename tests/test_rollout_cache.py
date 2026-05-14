"""Tests for RolloutCache — ring buffer keyed by (step, uid)."""
from __future__ import annotations

import pytest
import torch

from src.rac.rollout_cache import CachedRolloutBatch, RolloutCache


def _make_batch(step: int, uids: list[str], response_len: int = 4) -> CachedRolloutBatch:
    B = len(uids)
    return CachedRolloutBatch(
        step=step,
        uids=uids,
        prompt_ids=torch.randint(0, 100, (B, 8)),
        response_ids=torch.randint(0, 100, (B, response_len)),
        attention_mask=torch.ones(B, 8 + response_len, dtype=torch.long),
        response_mask=torch.ones(B, response_len, dtype=torch.long),
        old_log_probs=torch.randn(B, response_len),
        A_partial=torch.randn(B, response_len),
        r_fast=torch.randn(B),
    )


def test_stash_and_get_roundtrip():
    cache = RolloutCache(max_ttl_steps=10)
    b = _make_batch(step=5, uids=["a", "b", "c"])
    cache.stash(5, b)
    assert len(cache) == 3
    assert cache.n_batches == 1
    assert (5, "b") in cache

    got = cache.get_batch(5, ["b", "c"])
    assert got is not None
    assert got.uids == ["b", "c"]
    assert torch.equal(got.r_fast, b.r_fast[[1, 2]])


def test_get_nonexistent_returns_none():
    cache = RolloutCache(max_ttl_steps=10)
    cache.stash(5, _make_batch(5, ["a"]))
    assert cache.get_batch(999, ["a"]) is None
    assert cache.get_batch(5, ["missing"]) is None


def test_drain_old_evicts_below_ttl():
    cache = RolloutCache(max_ttl_steps=5)
    cache.stash(0, _make_batch(0, ["u0"]))
    cache.stash(3, _make_batch(3, ["u1"]))
    cache.stash(8, _make_batch(8, ["u2"]))
    assert len(cache) == 3

    n_removed = cache.drain_old(current_step=10)
    # cutoff = 10 - 5 = 5; step 0 and 3 evicted; step 8 kept
    assert n_removed == 2
    assert len(cache) == 1
    assert (0, "u0") not in cache
    assert (3, "u1") not in cache
    assert (8, "u2") in cache


def test_stash_step_mismatch_raises():
    cache = RolloutCache(max_ttl_steps=10)
    b = _make_batch(step=5, uids=["a"])
    with pytest.raises(ValueError, match="batch.step"):
        cache.stash(7, b)


def test_ttl_zero_raises():
    with pytest.raises(ValueError, match="max_ttl_steps"):
        RolloutCache(max_ttl_steps=0)


def test_select_slicing_preserves_alignment():
    """CachedRolloutBatch.select must keep rows aligned across all tensors."""
    b = _make_batch(step=5, uids=["a", "b", "c", "d"])
    sub = b.select([1, 3])
    assert sub.uids == ["b", "d"]
    assert torch.equal(sub.r_fast, b.r_fast[[1, 3]])
    assert torch.equal(sub.old_log_probs, b.old_log_probs[[1, 3]])
    assert sub.prompt_ids.shape[0] == 2


def test_to_dataproto_fallback_path():
    """When verl absent, to_dataproto returns a dict structure."""
    b = _make_batch(step=5, uids=["a", "b"])
    out = b.to_dataproto()
    # Either a verl DataProto or the dict fallback
    assert hasattr(out, "batch") or ("batch" in out and "non_tensor_batch" in out)


def test_multiple_steps_independent_uid_indexing():
    cache = RolloutCache(max_ttl_steps=100)
    cache.stash(5, _make_batch(5, ["same_uid"]))
    cache.stash(6, _make_batch(6, ["same_uid"]))
    # Same uid at different steps → independent lookups
    assert (5, "same_uid") in cache
    assert (6, "same_uid") in cache
    got5 = cache.get_batch(5, ["same_uid"])
    got6 = cache.get_batch(6, ["same_uid"])
    assert got5.step == 5
    assert got6.step == 6
