from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..games import Game
from ..states import State, TMove


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

    def predict_parallel(self, state: list[State]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        ...

    def set_to_eval(self) -> None:
        ...

    def train(self, training_set: list[TrainingData], minibatch_size: int) -> None:
        ...

    def save_training_data(self, training_set: list[TrainingData]) -> None:
        ...

    def save(self) -> None:
        ...


class DummyNeuralNetwork:
    def predict(self, state: State[TMove]) -> tuple[NDArray[np.float32], float]:
        policy: NDArray[np.float32] = np.ones(state.action_size, dtype=np.float32)
        return policy / policy.sum(), 0.0

    def set_to_eval(self) -> None:
        pass

    def train(self, training_set: list[TrainingData]) -> None:
        ...

    def save(self, generation: int) -> None:
        ...
