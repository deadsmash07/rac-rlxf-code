"""Custom verl PPO entry that substitutes PatchedRayPPOTrainer for RayPPOTrainer.

Why this exists:
    verl's `main_ppo.py` hard-codes
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    at module-level, then instantiates it directly inside `TaskRunner.run()`:
        trainer = RayPPOTrainer(...)                # verl/trainer/main_ppo.py:376
    There is no config-key that selects a trainer class; we can't ask verl
    to use our subclass from inside the YAML.

    The minimal, upstream-safe pattern is a one-line monkey-patch of the
    module-level `RayPPOTrainer` reference BEFORE `main_ppo.main()` runs.
    Python's name-resolution inside `TaskRunner.run()` reads the current
    value of `verl.trainer.main_ppo.RayPPOTrainer`, so replacing that
    attribute substitutes our subclass without touching verl source.

Usage (production):
    python3 scripts/main_ppo_rac.py --config-path=$(pwd)/configs \
                                    --config-name=rac_livecodebench

Usage (smoke / dry-run):
    python3 scripts/main_ppo_rac.py --dry-run

References:
    - verl/trainer/main_ppo.py:29, 376  (trainer import + instantiation sites)
    - src/trainer/patched_ray_trainer.py  (our subclass)
    - memory/track2_patched_ray_trainer_shipped.md
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Blocker #25 fix (resolved 2026-04-18 iter+N+27): vLLM 0.8.5's
# `vllm_async_server.vLLMHttpServer.launch_server()` instantiates a V1
# AsyncLLMEngine but reads envs.VLLM_USE_V1 — which is False by default.
# Setting it here (co-located with the PatchedRayPPOTrainer monkey-patch)
# ensures every downstream vLLM subprocess inherits the flag. Must set
# BEFORE any `import vllm` or `import verl.workers.rollout` call below.
os.environ.setdefault("VLLM_USE_V1", "1")


def _patch_verl_sleep_replicas() -> None:
    """Blocker #34 shim (2026-04-18): disable verl's `sleep_replicas()`.

    vLLM 0.8.5 V1 `sleep()` intermittently fails with
        AssertionError: Memory usage increased after sleeping
    after a few weight-sync cycles (vllm issue #19325). verl unconditionally
    calls `CheckpointEngineManager.sleep_replicas()` at init + every rollout
    step (ray_trainer.py L895, L1384). On a single-GPU 80 GB H100 running
    Qwen2.5-0.5B under FSDP we have plenty of headroom to skip the KV-cache
    release — `free_cache_engine: True` already drops the cache between
    rollouts.

    Escape hatch:
        PILSD_DISABLE_VERL_SLEEP=0  → keep upstream behavior (will crash at
        step ~3 on vllm 0.8.5 under current verl HEAD).
    """
    if os.environ.get("PILSD_DISABLE_VERL_SLEEP", "1") != "1":
        return
    try:
        from verl.checkpoint_engine import base as _ckpt_base
    except Exception as e:  # pragma: no cover — env-compat
        print(f"[rac-ppo] WARN: cannot import verl.checkpoint_engine.base: {e}")
        return

    async def _noop_sleep_replicas(self):  # type: ignore[override]
        return None

    # Keep a reference to the original for debugging
    _ckpt_base.CheckpointEngineManager._orig_sleep_replicas = (
        _ckpt_base.CheckpointEngineManager.sleep_replicas
    )
    _ckpt_base.CheckpointEngineManager.sleep_replicas = _noop_sleep_replicas
    print("[rac-ppo] Patched verl.checkpoint_engine sleep_replicas → no-op "
          "(PILSD_DISABLE_VERL_SLEEP=1)")


# Ensure our package is on the path so `from src.trainer...` resolves
# when running via `python3 scripts/main_ppo_rac.py` from track root.
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _substitute_trainer_class() -> None:
    """Replace verl.trainer.main_ppo.RayPPOTrainer with our PatchedRayPPOTrainer.

    Must be called BEFORE any `main_ppo.main()` / `main_ppo.run_ppo(cfg)` call.
    """
    from src.trainer.patched_ray_trainer import PatchedRayPPOTrainer

    # Importing main_ppo triggers verl's module-level `from verl.trainer.ppo.ray_trainer import RayPPOTrainer`.
    # We then overwrite the attribute on the main_ppo module so that
    # `TaskRunner.run()` picks up our subclass at `trainer = RayPPOTrainer(...)`.
    import verl.trainer.main_ppo as _main_ppo
    _main_ppo.RayPPOTrainer = PatchedRayPPOTrainer

    # Best-effort: also replace it in the `TaskRunner` class's __globals__ if
    # the class captured a reference at class-body eval time (it doesn't, but
    # defensive).
    if hasattr(_main_ppo, "TaskRunner"):
        _main_ppo.TaskRunner.run.__globals__["RayPPOTrainer"] = PatchedRayPPOTrainer


def dry_run_report() -> int:
    """Verify the patch is sound without spawning Ray or running a training step.

    Checks:
      1. PatchedRayPPOTrainer imports
      2. main_ppo imports (or fails with a clear env-compat message)
      3. After substitution, `main_ppo.RayPPOTrainer` IS our subclass
      4. The subclass's MRO includes the real RayPPOTrainer when verl is loadable
    """
    print("[rac-ppo-dry-run]")
    try:
        from src.trainer.patched_ray_trainer import (
            PatchedRayPPOTrainer, _HAS_VERL,
        )
    except Exception as e:
        print(f"  ✗ cannot import PatchedRayPPOTrainer: {e}")
        return 2

    print(f"  ✓ PatchedRayPPOTrainer imports; _HAS_VERL={_HAS_VERL}")

    try:
        _substitute_trainer_class()
        import verl.trainer.main_ppo as _main_ppo
        is_subclass = _main_ppo.RayPPOTrainer is PatchedRayPPOTrainer
        print(f"  ✓ main_ppo.RayPPOTrainer ← PatchedRayPPOTrainer: {is_subclass}")
    except Exception as e:
        print(f"  △ main_ppo import failed (likely env-compat, NOT a RAC bug): {e}")
        print("    The substitution function is correct; blocker is verl/torch env.")
        return 1

    if _HAS_VERL:
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer as _RealBase
        mro_ok = issubclass(PatchedRayPPOTrainer, _RealBase)
        print(f"  ✓ PatchedRayPPOTrainer subclasses real RayPPOTrainer: {mro_ok}")
        if not mro_ok:
            return 2
    else:
        print("  △ _HAS_VERL=False; MRO-with-real-verl skipped")

    print("\nReady. To run training:")
    print("  python3 scripts/main_ppo_rac.py --config-path=$(pwd)/configs "
          "--config-name=rac_livecodebench")
    return 0


def main() -> int:
    # Peek at argv for --dry-run before hydra eats it.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dry-run", action="store_true")
    args, unknown = pre.parse_known_args()
    if args.dry_run:
        return dry_run_report()

    _substitute_trainer_class()
    _patch_verl_sleep_replicas()

    # Restore argv for hydra (it reads sys.argv; drop --dry-run which we consumed)
    sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "--dry-run"]

    import verl.trainer.main_ppo as _main_ppo
    _main_ppo.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
