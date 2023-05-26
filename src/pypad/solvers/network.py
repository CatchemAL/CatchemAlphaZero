from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..states import State, TMove
from ..games import Game


@dataclass
class TrainingData:
    encoded_state: np.ndarray
    policy: np.ndarray
    outcome: float


class NeuralNetwork(Protocol):
    @property
    def game(self) -> Game:
        ...

    @property
    def action_size(self) -> int:
        ...

    @property
    def generation(self) -> int:
        ...

    def predict(self, state: State[TMove]) -> tuple[NDArray[np.float32], float]:
        ...

    def set_to_eval(self) -> None:
        ...

    def set_to_train(self) -> None:
        ...

    def train(self, training_set: list[TrainingData]) -> None:
        ...

    def save(self) -> None:
        ...


class DummyNeuralNetwork:
    def predict(self, state: State[TMove]) -> tuple[NDArray[np.float32], float]:
        policy: NDArray[np.float32] = np.ones(state.action_size, dtype=np.float32)
        return policy / policy.sum(), 0.0

    def set_to_eval(self) -> None:
        pass

    def set_to_train(self) -> None:
        pass

    def train(self, training_set: list[TrainingData]) -> None:
        ...

    def save(self, generation: int) -> None:
        ...
