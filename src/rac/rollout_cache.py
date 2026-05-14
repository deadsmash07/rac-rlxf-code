"""Ring-buffer cache for past rollouts awaiting slow-channel reward arrival.

Needed because verl recomputes `old_log_probs` fresh each step
(`ray_trainer.py:1185-1215`) and discards the batch right after
`_update_actor` — we have nothing to reference at correction time unless
we stash it ourselves. Per `src/verl_integration_notes.md` §3.

API:
  cache = RolloutCache(max_ttl_steps=50)
  cache.stash(step, batch)                    # trainer hook after L1494
  cached = cache.get_batch(step_t, uids)       # advantage corrector reads
  cache.drain_old(current_step)                # evicts entries older than TTL

References:
  - Espeholt et al. 2018 IMPALA — FIFO replay pattern for off-policy IS
  - Mnih et al. 2015 DQN — experience-replay ring buffer
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class CachedRolloutBatch:
    """Subset of a verl DataProto, the fields RAC needs at correction time.

    `uids` and the tensor fields are aligned by row index: `uids[i]` describes
    `old_log_probs[i]`, `response_mask[i]`, `A_partial[i]`.
    """

    step: int                    # optimizer step when these were produced
    uids: list[str]
    prompt_ids: torch.Tensor     # (B, prompt_len)
    response_ids: torch.Tensor   # (B, response_len)
    attention_mask: torch.Tensor # (B, total_len)
    response_mask: torch.Tensor  # (B, response_len) — 1 for real tokens, 0 for padding
    old_log_probs: torch.Tensor  # (B, response_len) — log π_{θ(t)}(a|s) per token
    A_partial: torch.Tensor      # (B, response_len) — fast-channel partial advantage
    r_fast: torch.Tensor         # (B,) — scalar fast-channel reward per rollout

    def select(self, indices: list[int]) -> "CachedRolloutBatch":
        """Return a new CachedRolloutBatch restricted to rows at `indices`."""
        return CachedRolloutBatch(
            step=self.step,
            uids=[self.uids[i] for i in indices],
            prompt_ids=self.prompt_ids[indices],
            response_ids=self.response_ids[indices],
            attention_mask=self.attention_mask[indices],
            response_mask=self.response_mask[indices],
            old_log_probs=self.old_log_probs[indices],
            A_partial=self.A_partial[indices],
            r_fast=self.r_fast[indices],
        )

    def to_dataproto(self) -> Any:
        """Best-effort conversion to verl's DataProto for log-prob recomputation.

        Used by `apply_rac_correction` to feed `actor_rollout_wg.compute_log_prob`.
        Falls back to a dict if verl is not installed (so tests can still run).
        """
        batch_dict = {
            "prompts": self.prompt_ids,
            "responses": self.response_ids,
            "attention_mask": self.attention_mask,
            "response_mask": self.response_mask,
            "old_log_probs": self.old_log_probs,
        }
        try:
            import numpy as np
            from verl import DataProto  # type: ignore[import-not-found]
            from tensordict import TensorDict
            td = TensorDict(batch_dict, batch_size=[len(self.uids)])
            # verl.protocol enforces non_tensor_batch values be np.ndarray, not list.
            # See verl/protocol.py:463 assert isinstance(val, np.ndarray).
            uid_arr = np.array(list(self.uids), dtype=object)
            return DataProto(batch=td, non_tensor_batch={"uid": uid_arr})
        except ImportError:
            # Pure-dict fallback for tests without verl
            return {"batch": batch_dict, "non_tensor_batch": {"uid": list(self.uids)}}


class RolloutCache:
    """Ring buffer of `CachedRolloutBatch`s keyed by (step, uid).

    Entries are evicted when either (a) their step is older than
    `current_step - max_ttl_steps`, or (b) the total entry count exceeds
    `capacity_bytes_budget` (not implemented yet — policy is step-TTL only).
    """

    def __init__(self, max_ttl_steps: int = 50):
        if max_ttl_steps < 1:
            raise ValueError("max_ttl_steps must be >= 1")
        self.max_ttl_steps = max_ttl_steps
        # OrderedDict keyed by step — allows O(log n) range evictions
        self._by_step: OrderedDict[int, CachedRolloutBatch] = OrderedDict()
        # Flat lookup (step, uid) → row index in that step's batch
        self._uid_index: dict[tuple[int, str], int] = {}

    def stash(self, step: int, batch: CachedRolloutBatch) -> None:
        """Store `batch` under `step` key. Called by trainer hook after
        `old_log_probs` are computed."""
        if batch.step != step:
            raise ValueError(
                f"stash: batch.step={batch.step} != step arg={step}"
            )
        self._by_step[step] = batch
        for i, uid in enumerate(batch.uids):
            self._uid_index[(step, uid)] = i

    def get_batch(
        self, step_t: int, uids: list[str]
    ) -> Optional[CachedRolloutBatch]:
        """Retrieve rows matching `uids` from the batch stashed at `step_t`.

        Returns None if no batch is cached for that step or no uids match.
        """
        if step_t not in self._by_step:
            return None
        cached = self._by_step[step_t]
        indices = []
        for u in uids:
            idx = self._uid_index.get((step_t, u))
            if idx is not None:
                indices.append(idx)
        if not indices:
            return None
        return cached.select(indices)

    def drain_old(self, current_step: int) -> int:
        """Evict all batches with `step < current_step - max_ttl_steps`.

        Returns the number of entries removed.
        """
        cutoff = current_step - self.max_ttl_steps
        to_remove = [s for s in self._by_step.keys() if s < cutoff]
        for s in to_remove:
            batch = self._by_step.pop(s)
            for uid in batch.uids:
                self._uid_index.pop((s, uid), None)
        return len(to_remove)

    def __len__(self) -> int:
        return sum(len(b.uids) for b in self._by_step.values())

    def __contains__(self, key: tuple[int, str]) -> bool:
        return key in self._uid_index

    @property
    def n_batches(self) -> int:
        return len(self._by_step)
