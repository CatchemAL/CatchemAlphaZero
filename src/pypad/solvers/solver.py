from abc import ABC, abstractmethod

from ..games import State
from ..games.state import TMove


class Solver(ABC):
    @abstractmethod
    def solve(self, state: State[TMove]) -> TMove:
        ...
