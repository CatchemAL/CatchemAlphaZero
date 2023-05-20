from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Generator, Generic, Tuple, TypeVar

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
    def shape(self) -> Tuple[int, int]:
        ...

    @abstractmethod
    def legal_moves(self) -> Generator[TMove, None, None]:
        ...

    @abstractmethod
    def play_move(self, move: TMove) -> None:
        ...

    @abstractmethod
    def is_won(self) -> bool:
        ...

    @abstractmethod
    def outcome(self, perspective: int, indicator: str = "win-loss") -> float:
        ...

    @abstractmethod
    def html(self, is_tiny_repr: bool = False) -> str:
        ...

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        ...

    @abstractmethod
    def __copy__(self) -> "State[TMove]":
        ...
