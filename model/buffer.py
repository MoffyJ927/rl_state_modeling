"""RolloutBuffer: PPO experience storage for one episode trajectory."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RolloutBuffer:
    """Stores one rollout trajectory."""
    states: List[Dict] = field(default_factory=list)
    actions: List[Dict] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def clear(self):
        for v in vars(self).values():
            if isinstance(v, list):
                v.clear()

    def size(self) -> int:
        return len(self.states)
