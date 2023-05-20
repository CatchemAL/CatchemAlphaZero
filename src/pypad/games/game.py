from abc import ABC, abstractmethod
from typing import Generic

import numpy as np

from ..kaggle_types import Configuration, Observation
from ..states import ConnectXState, TicTacToeState, TState
from ..views import View
from ..views.console import ConsoleConnectXView, ConsoleTicTacToeView


class Game(ABC, Generic[TState]):
    @abstractmethod
    def initial_state(self, start: str) -> TState:
        ...

    @abstractmethod
    def from_kaggle(self, obs: Observation, config: Configuration) -> TState:
        ...

    @abstractmethod
    def display(self, state: TState) -> None:
        ...

    @abstractmethod
    def display_outcome(self, state: TState) -> None:
        ...


class ConnectX(Game[ConnectXState]):
    def __init__(self, rows: int = 6, cols: int = 7, view: View[ConnectXState] | None = None) -> None:
        self.rows = rows
        self.cols = cols
        self.view = view or ConsoleConnectXView()

    def initial_state(self, start: str) -> ConnectXState:
        return ConnectXState.create(self.rows, self.cols)

    def from_kaggle(self, obs: Observation, config: Configuration) -> ConnectXState:
        grid = np.asarray(obs.board).reshape(config.rows, config.columns)
        state = ConnectXState.from_grid(grid)
        return state

    def display(self, state: ConnectXState) -> None:
        self.view.display(state)

    def display_outcome(self, state: ConnectXState) -> None:
        self.view.display_outcome(state)


class TicTacToe(Game[TicTacToeState]):
    def __init__(self, view: View[TicTacToeState] | None = None) -> None:
        self.view = view or ConsoleTicTacToeView()

    def initial_state(self, start: str) -> TicTacToeState:
        return TicTacToeState.create()

    def from_kaggle(self, obs: Observation, config: Configuration) -> TicTacToeState:
        grid = np.asarray(obs.board).reshape(3, 3)
        state = TicTacToeState.from_grid(grid)
        return state

    def display(self, state: TicTacToeState) -> None:
        self.view.display(state)

    def display_outcome(self, state: ConnectXState) -> None:
        self.view.display_outcome(state)


class Chess:
    pass
