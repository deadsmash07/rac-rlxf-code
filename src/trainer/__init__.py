"""verl-plugin reward manager + trainer wrappers for Track 2."""

from .multi_channel_reward_manager import (
    MultiChannelRACRewardManager,
    PendingRollout,
    has_verl,
)

__all__ = ["MultiChannelRACRewardManager", "PendingRollout", "has_verl"]
