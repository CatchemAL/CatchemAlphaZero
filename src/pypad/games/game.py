from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic

import numpy as np

from ..kaggle_types import Configuration, Observation
from ..solvers import Solver
from ..states import ConnectXState, TicTacToeState, TState
from ..views import View
from ..views.console import ConsoleConnectXView, ConsoleTicTacToeView
from .game_type import GameType


@dataclass(frozen=True, slots=True)
class GameParameters:
    shape: tuple[int, int]
    observation_shape: tuple[int, int, int]
    action_size: int


class Game(ABC, Generic[TState]):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def fullname(self) -> str:
        ...

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        ...

    @abstractmethod
    def initial_state(self, start: str | list[int] | None = None) -> TState:
        ...

    @abstractmethod
    def config(self) -> GameParameters:
        ...

    @abstractmethod
    def from_kaggle(self, obs: Observation, config: Configuration) -> TState:
        ...

    def create_agent(self, player: Solver) -> Callable[[Observation, Configuration], int]:
        def get_best_move(obs: Observation, config: Configuration) -> int:
            state = self.from_kaggle(obs, config)
            move = player.solve(state)
            return move

        return get_best_move

    @abstractmethod
    def symmetries(
        self, encoded_state: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
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

    @property
    def name(self) -> str:
        return "connectx"

    @property
    def fullname(self) -> str:
        return f"ConnectX_{self.rows}x{self.cols}"

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    def initial_state(self, start: str | list[int] | None = None) -> ConnectXState:
        return ConnectXState.create(self.rows, self.cols, start)

    def config(self) -> GameParameters:
        NUM_PLANES = 3

        shape = self.shape
        observation_shape = self.rows, self.cols, NUM_PLANES
        action_size = self.cols
        return GameParameters(shape, observation_shape, action_size)

    def from_kaggle(self, obs: Observation, config: Configuration) -> ConnectXState:
        grid = np.asarray(obs.board).reshape(config.rows, config.columns)
        state = ConnectXState.from_grid(grid)
        return state

    def display(self, state: ConnectXState) -> None:
        self.view.display(state)

    def display_outcome(self, state: ConnectXState) -> None:
        self.view.display_outcome(state)


class TicTacToe(Game[TicTacToeState]):
    ROWS, COLS = 3, 3

    def __init__(self, view: View[TicTacToeState] | None = None) -> None:
        self.view = view or ConsoleTicTacToeView()

    @property
    def name(self) -> str:
        return "tictactoe"

    @property
    def fullname(self) -> str:
        return f"TicTacToe"

    @property
    def shape(self) -> tuple[int, int]:
        return self.ROWS, self.COLS

    def initial_state(self, start: str | list[int] | None = None) -> TicTacToeState:
        return TicTacToeState.create(start)

    def config(self) -> GameParameters:
        NUM_PLANES = 3

        shape = self.shape
        observation_shape = self.ROWS, self.COLS, NUM_PLANES
        action_size = self.ROWS * self.COLS
        return GameParameters(shape, observation_shape, action_size)

    def from_kaggle(self, obs: Observation, config: Configuration) -> TicTacToeState:
        grid = np.asarray(obs.board).reshape(self.ROWS, self.COLS)
        state = TicTacToeState.from_grid(grid)
        return state

    def symmetries(
        self, encoded_state: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        syms: list[tuple[np.ndarray, np.ndarray]] = []
        for rot in range(4):
            state = np.rot90(encoded_state, axes=(1, 2), k=rot)
            pol = np.rot90(policy.reshape(self.ROWS, self.COLS), k=rot).flatten()
            syms.append((state, pol))

        flipped_state = encoded_state[:, :, ::-1]
        flipped_policy = np.fliplr(policy.reshape(3, 3)).flatten()

        for rot in range(4):
            state = np.rot90(flipped_state, axes=(1, 2), k=rot)
            pol = np.rot90(flipped_policy.reshape(self.ROWS, self.COLS), k=rot).flatten()
            syms.append((state, pol))

        return syms

    def display(self, state: TicTacToeState) -> None:
        self.view.display(state)

    def display_outcome(self, state: TicTacToeState) -> None:
        self.view.display_outcome(state)


class Chess:
    pass


def get_game(game_type: GameType) -> TicTacToe | ConnectX:
    match game_type:
        case GameType.TICTACTOE:
            return TicTacToe()
        case GameType.CONNECTX:
            return ConnectX()
        case _:
            raise ValueError(f"Invalid game type: {game_type}")
