from abc import ABC, abstractmethod
from enum import Enum
from typing import TypeVar, Generic, Iterable

TMove = TypeVar("TMove")


class State(ABC, Generic[TMove]):
    @abstractmethod
    def play_move(self, move: TMove) -> None:
        pass

    @abstractmethod
    def legal_moves(self) -> Iterable[TMove]:
        pass


class ChessState(State[str]):
    def play_move(self, move: str) -> None:
        print(f"Playing move {move} in chess...")

    def legal_moves(self) -> Iterable[str]:
        return ["e2e4", "d2d4"]


class TicTacToeState(State[int]):
    def play_move(self, move: int) -> None:
        print(f"Playing move {move} in tic-tac-toe...")

    def legal_moves(self) -> Iterable[int]:
        return [0, 1, 2, 3, 4, 5, 6, 7, 8]


class Solver(Generic[TMove], ABC):
    @abstractmethod
    def solve(self, board_state: State[TMove]) -> TMove:
        ...


class MinimaxSolver(Solver[TMove]):
    def solve(self, board_state: State[TMove]) -> TMove:
        # Minimax implementation here
        raise NotImplementedError()


class MCTSSolver(Solver[TMove]):
    def solve(self, board_state: State[TMove]) -> TMove:
        # MCTS implementation here
        raise NotImplementedError()


class SolverType(Enum):
    MINIMAX = MinimaxSolver
    MCTS = MCTSSolver


class GameType(Enum):
    CHESS = "chess"
    TIC_TAC_TOE = "tic-tac-toe"


def create_state(game_type: GameType) -> State[TMove]:
    if game_type == GameType.CHESS:
        return ChessState()
    elif game_type == GameType.TIC_TAC_TOE:
        return TicTacToeState()
    else:
        raise ValueError(f"Unsupported game type: {game_type}")


def create_solver(solver_type: SolverType, **kwargs) -> Solver[TMove]:
    solver_class = solver_type.value
    return solver_class(**kwargs)


def bootstrap(solver_type: SolverType, game_type: GameType) -> None:
    state = create_state(game_type)
    solver = create_solver(solver_type)
    next_move = solver.solve(state)
    state.play_move(next_move)
