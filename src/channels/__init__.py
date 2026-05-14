"""Reward channel implementations for MultiChannelRACRewardManager.

Registry pattern: each channel class registers itself via @register_channel
so YAML configs can reference channels by `type: <name>` and we resolve to
the actual callable.
"""

from .registry import (
    register_channel,
    get_channel_class,
    list_registered,
    build_channel_from_config,
)
from .code_exec import CodeExecChannel
from .livecodebench_judge import LiveCodeBenchJudgeChannel
from .mock_channels import MockSyncChannel, MockAsyncChannel
from .gsm8k_channels import GSM8KFastChannel, GSM8KDelayedChannel
from .pilsd_reward_channels import PILSDFastChannel, PILSDDelayedChannel

__all__ = [
    "register_channel",
    "get_channel_class",
    "list_registered",
    "build_channel_from_config",
    "CodeExecChannel",
    "LiveCodeBenchJudgeChannel",
    "MockSyncChannel",
    "MockAsyncChannel",
    "GSM8KFastChannel",
    "GSM8KDelayedChannel",
    "PILSDFastChannel",
    "PILSDDelayedChannel",
]
