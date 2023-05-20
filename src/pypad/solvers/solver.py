from abc import ABC, abstractmethod

from ..states import State, TMove


class Solver(ABC):
    @abstractmethod
    def solve(self, state: State[TMove]) -> TMove:
        ...
