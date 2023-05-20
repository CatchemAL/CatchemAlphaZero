from abc import ABC, abstractmethod
from typing import Generic

from ..states import TState


class View(ABC, Generic[TState]):
    @abstractmethod
    def display(self, state: TState) -> None:
        ...
