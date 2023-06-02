from dataclasses import dataclass, field

import numpy as np

from ..states import State


@dataclass(slots=True)
class RecordedAction:
    state_before: State
    policy: np.ndarray
    move: int
    state_after: State


@dataclass(slots=True)
class ParallelGame:
    initial_state: State
    recorded_actions: list[RecordedAction] = field(default_factory=list)

    @property
    def latest_state(self) -> State:
        if self.recorded_actions:
            return self.recorded_actions[-1].state_after
        return self.initial_state
