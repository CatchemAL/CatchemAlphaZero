from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic

import numpy as np

from ..kaggle_types import Configuration, Observation
from ..solvers import Solver
from ..states import ChessState, ConnectXState, TicTacToeState, TState
from ..states.chess_enums import ActionPlanes, ObsPlanes
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

    def generate_states(
        self, depth: int, min_num_states: int = 1, start: str | list[int] | None = None
    ) -> list[TState]:
        state = self.initial_state(start)
        states = Game._get_all_states(state, depth)
        for _ in range(len(states), min_num_states):
            states.append(self.initial_state(start))

        return states

    @staticmethod
    def _get_all_states(state: TState, depth: int) -> list[TState]:
        all_states = [state]
        if depth == 0:
            return all_states

        legal_moves = state.status().legal_moves
        for move in legal_moves:
            new_state = state.play_move(move)
            states_at_next_depth = Game._get_all_states(new_state, depth - 1)
            all_states += states_at_next_depth

        return all_states


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
        observation_shape = NUM_PLANES, self.rows, self.cols
        action_size = self.cols
        return GameParameters(shape, observation_shape, action_size)

    def from_kaggle(self, obs: Observation, config: Configuration) -> ConnectXState:
        grid = np.asarray(obs.board).reshape(config.rows, config.columns)
        state = ConnectXState.from_grid(grid)
        return state

    def symmetries(
        self, encoded_state: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        syms: list[tuple[np.ndarray, np.ndarray]] = []

        flipped_state = encoded_state[:, :, ::-1]
        flipped_policy = policy[::-1]

        syms.append((encoded_state, policy))
        syms.append((flipped_state, flipped_policy))

        return syms

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
        observation_shape = NUM_PLANES, self.ROWS, self.COLS
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


class Chess(Game[ChessState]):
    ROWS, COLS = 8, 8

    def __init__(self, view: View[ChessState] | None = None) -> None:
        self.view = view

    @property
    def name(self) -> str:
        return "chess"

    @property
    def fullname(self) -> str:
        return "Chess"

    @property
    def shape(self) -> tuple[int, int]:
        return self.ROWS, self.COLS

    def initial_state(self, start: str | list[int] | None = None) -> ChessState:
        return ChessState.create(start)

    def config(self) -> GameParameters:
        shape = self.shape
        return GameParameters(shape, ObsPlanes.shape(), ActionPlanes.size())

    def from_kaggle(self, obs: Observation, config: Configuration) -> ChessState:
        pass

    def symmetries(
        self, encoded_state: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(encoded_state, policy)]

    def display(self, state: ChessState) -> None:
        pass

    def display_outcome(self, state: ChessState) -> None:
        pass


def get_game(game_type: GameType) -> Chess | ConnectX | TicTacToe:
    match game_type:
        case GameType.TICTACTOE:
            return TicTacToe()
        case GameType.CONNECTX:
            return ConnectX()
        case GameType.CHESS:
            return Chess()
        case _:
            raise ValueError(f"Invalid game type: {game_type}")
