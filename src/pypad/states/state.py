from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

import numpy as np

TMove = TypeVar("TMove")
TState = TypeVar("TState", bound="State[+TMove]")
TState_co = TypeVar("TState_co", bound="State[+TMove]", covariant=True)


@dataclass
class Policy(Generic[TMove]):
    moves: list[TMove] | None
    priors: np.ndarray[np.float32] | None
    encoded_policy: np.ndarray[np.float32]
    value: float

    def select_move(self, temperature: float) -> TMove:
        if temperature <= 0.001:
            idx = np.argmax(self.priors)
        else:
            temperature_policy = self.priors ** (1 / temperature)
            temperature_policy /= temperature_policy.sum()
            idx = np.random.choice(len(self.priors), p=temperature_policy)

        return self.moves[idx]


class State(ABC, Generic[TMove]):
    @property
    @abstractmethod
    def played_by(self) -> int:
        ...

    @property
    @abstractmethod
    def move_count(self) -> int:
        ...

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        ...

    @abstractmethod
    def status(self) -> Status[TMove]:
        ...

    @abstractmethod
    def play_move(self, move: TMove) -> Self:
        ...

    @abstractmethod
    def set_move(self, move: TMove) -> None:
        ...

    @abstractmethod
    def policy_loc(self, move: TMove) -> int:
        ...

    @abstractmethod
    def get_input_move(self) -> TMove:
        ...

    @abstractmethod
    def html(self, is_tiny_repr: bool = False) -> str:
        ...

    @abstractmethod
    def to_feature(self) -> np.ndarray:
        ...

    @abstractmethod
    def __copy__(self) -> Self:
        ...


@dataclass
class TemperatureSchedule:
    cutoff: int
    temperature: float

    def get_temperature(self, move_count: int) -> float:
        return self.temperature if move_count < self.cutoff else 0

    @classmethod
    def competitive(cls) -> Self:
        return cls(0, 0)


@dataclass
class Status(Generic[TMove]):
    is_in_progress: bool
    played_by: int
    value: float
    legal_moves: list[TMove]

    def outcome(self, perspective: int) -> float:
        return self.value if perspective == self.played_by else -self.value
