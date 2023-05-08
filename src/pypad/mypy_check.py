from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generator, Generic, Iterable, Protocol, Tuple, TypeVar

import numpy as np
import numpy.typing as npt

TMove = TypeVar("TMove")
TMove_co = TypeVar("TMove_co", covariant=True)
TState = TypeVar("TState", bound="State[+TMove]")
TState_co = TypeVar("TState_co", bound="State[+TMove]", covariant=True)


class State2(Protocol[TMove]):
    @property
    def played_by(self) -> int:
        ...

    @property
    def shape(self) -> Tuple[int, int]:
        ...

    def legal_moves(self) -> Generator[TMove, None, None]:
        ...

    def play_move(self, move: TMove) -> None:
        ...

    def is_won(self) -> bool:
        ...

    def outcome(self, perspective: int, indicator: str = "win-loss") -> float:
        ...

    def to_numpy(self) -> npt.NDArray[np.float32]:
        ...

    def __copy__(self) -> "State[TMove]":
        ...


class State(ABC, Generic[TMove]):
    @abstractmethod
    def play_move(self, move: TMove) -> None:
        ...

    @abstractmethod
    def legal_moves(self) -> Iterable[TMove]:
        ...


class TicTacToe(State[int]):
    def legal_move(self) -> Iterable[int]:
        raise NotImplementedError()

    def play_move(self, move: int) -> None:
        raise NotImplementedError()


class StateFactory(ABC, Generic[TState_co]):
    @abstractmethod
    def load_initial_state(self, initial_position: str) -> TState_co:
        ...


class TicTacToeFactory(StateFactory[TicTacToe]):
    def load_initial_state(self, initial_position: str) -> TicTacToe:
        raise NotImplementedError()


class Solver(ABC):
    @abstractmethod
    def Solve(self, state: State[TMove]) -> TMove:
        ...


class MinimaxSolver(Solver):
    def Solve(self, state: State[TMove]) -> TMove:
        raise NotImplementedError()


class StateView(ABC, Generic[TState, TMove]):
    @abstractmethod
    def display(self, state: TState) -> None:
        pass


class TicTacToeView(StateView[TicTacToe, int]):
    def display(self, state: TicTacToe) -> None:
        raise NotImplementedError()


@dataclass
class Controller(Generic[TState, TMove]):
    state_factory: StateFactory[TState]
    solver: Solver
    view: StateView[TState, TMove]

    def Run(self) -> None:
        initial_state = self.state_factory.load_initial_state("placeholder")
        move = self.solver.Solve(initial_state)
        initial_state.play_move(move)
        self.view.display(initial_state)


def SampleCode() -> None:
    stateFactory = TicTacToeFactory()
    view = TicTacToeView()
    solver = MinimaxSolver()
    controller = Controller[TicTacToe, int](stateFactory, solver, view)
    controller.Run()


SampleCode()
