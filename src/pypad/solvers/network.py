from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..states import State, TMove


class NeuralNetwork(Protocol):
    def predict(self, state: State[TMove]) -> tuple[NDArray[np.float32], float]:
        ...


class DummyNeuralNetwork:
    def predict(self, state: State[TMove]) -> tuple[NDArray[np.float32], float]:
        policy: NDArray[np.float32] = np.ones(state.action_size, dtype=np.float32)
        return policy / policy.sum(), 0.0
