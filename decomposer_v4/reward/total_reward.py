from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DualReward:
    planner_reward: float
    answer_reward: float
    execution_reward: float = 0.0
    synthesis_reward: float = 0.0
