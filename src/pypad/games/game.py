from abc import ABC, abstractmethod
from typing import Callable, Generic

import numpy as np

from ..kaggle_types import Configuration, Observation
from ..solvers import Solver
from ..states import ConnectXState, TicTacToeState, TState
from ..views import View
from ..views.console import ConsoleConnectXView, ConsoleTicTacToeView


class Game(ABC, Generic[TState]):
    @abstractmethod
    def initial_state(self, start: str | list[int]) -> TState:
        ...

    @abstractmethod
    def from_kaggle(self, obs: Observation, config: Configuration) -> TState:
        ...

    @abstractmethod
    def to_kaggle_move(self, state: TState, move: int) -> int:
        ...

    @property
    @abstractmethod
    def label(self) -> str:
        ...

    def create_agent(self, player: Solver) -> Callable[[Observation, Configuration], int]:
        def get_best_move(obs: Observation, config: Configuration) -> int:
            state = self.from_kaggle(obs, config)
            move = player.solve(state)
            return self.to_kaggle_move(state, move)

        return get_best_move

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

    def initial_state(self, start: str | list[int] | None = None) -> ConnectXState:
        return ConnectXState.create(self.rows, self.cols, start)

    @property
    def label(self) -> str:
        return "connectx"

    def from_kaggle(self, obs: Observation, config: Configuration) -> ConnectXState:
        grid = np.asarray(obs.board).reshape(config.rows, config.columns)
        state = ConnectXState.from_grid(grid)
        return state

    def to_kaggle_move(self, state: ConnectXState, move: int) -> int:
        return state.bitboard_util.move_to_col(move)

    def display(self, state: ConnectXState) -> None:
        self.view.display(state)

    def display_outcome(self, state: ConnectXState) -> None:
        self.view.display_outcome(state)


class TicTacToe(Game[TicTacToeState]):
    def __init__(self, view: View[TicTacToeState] | None = None) -> None:
        self.view = view or ConsoleTicTacToeView()

    def initial_state(self, start: str | list[int] | None = None) -> TicTacToeState:
        return TicTacToeState.create(start)

    def from_kaggle(self, obs: Observation, config: Configuration) -> TicTacToeState:
        grid = np.asarray(obs.board).reshape(3, 3)
        state = TicTacToeState.from_grid(grid)
        return state

    def to_kaggle_move(self, _: ConnectXState, move: int) -> int:
        return move

    @property
    def label(self) -> str:
        return "tictactoe"

    def display(self, state: TicTacToeState) -> None:
        self.view.display(state)

    def display_outcome(self, state: ConnectXState) -> None:
        self.view.display_outcome(state)


class Chess:
    pass
