from typing import Any, Callable

from .agent_type import AgentType
from .controller import Controller
from .games import GameType
from .games.connectx import ConnectXFactory, ConnectXView
from .games.tictactoe import TicTacToeFactory, TicTacToeView
from .kaggle_types import Configuration, Observation
from .solvers import Solver


def load_player(player_type: AgentType) -> Solver:
    match player_type:
        case AgentType.HUMAN:
            raise ValueError("todo")
        case AgentType.MINIMAX:
            raise ValueError("todo")
        case AgentType.MCTS:
            from .solvers.mcts import MctsSolver

            return MctsSolver()
        case AgentType.AZ:
            raise ValueError("todo")
        case _:
            raise ValueError(f"Unsupported agent type: {player_type}")


def get_controller(game_type: GameType, player1: Solver, player2: Solver) -> Controller[Any, Any]:
    match game_type:
        case GameType.TICTACTOE:
            state_factory = TicTacToeFactory()
            view = TicTacToeView()
            return Controller(player1, player2, state_factory, view)
        case GameType.CONNECTX:
            state_factory = ConnectXFactory()
            view = ConnectXView()
            return Controller(player1, player2, state_factory, view)
        case _:
            raise ValueError(f"Unsupported game type: {game_type}")


def create_agent(game_type: GameType, player: Solver) -> Callable[[Observation, Configuration], int]:
    match game_type:
        case GameType.TICTACTOE:
            state_factory = TicTacToeFactory()
        case GameType.CONNECTX:
            state_factory = ConnectXFactory()
        case _:
            raise ValueError(f"Unsupported game type: {game_type}")

    def get_best_move(obs: Observation, config: Configuration) -> int:
        state = state_factory.from_kaggle(obs, config)
        move = player.solve(state)
        return move

    return get_best_move
