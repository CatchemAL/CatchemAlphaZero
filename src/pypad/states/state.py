from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

import numpy as np

TMove = TypeVar("TMove")
TState = TypeVar("TState", bound="State[+TMove]")
TState_co = TypeVar("TState_co", bound="State[+TMove]", covariant=True)


class State(ABC, Generic[TMove]):
    @property
    @abstractmethod
    def played_by(self) -> int:
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
    def select_move(self, policy: np.ndarray, schedule: TemperatureSchedule) -> TMove:
        ...

    @abstractmethod
    def is_won(self) -> bool:
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

    def get_temperature(self, move_count) -> float:
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
