from enum import Enum
from typing import Callable

from ..games import Game
from ..kaggle_types import Configuration, Observation
from ..states import TState
from .solver import Solver


class AgentType(Enum):
    """Enum representing the types of agents supported."""

    HUMAN = "HUMAN"
    MINIMAX = "MINIMAX"
    MCTS = "MCTS"
    AZ = "AZ"

    @staticmethod
    def from_str(value: str) -> "AgentType":
        """Converts a string to its enum representation.

        Args:
            value (str): The string representation of the agent type.

        Raises:
            ValueError: If the string is not recognised.

        Returns:
            AgentType: The enum
        """
        if value.upper() == "HUMAN":
            return AgentType.HUMAN
        if value.upper() == "MINIMAX":
            return AgentType.MINIMAX
        if value.upper() == "MCTS":
            return AgentType.MCTS
        if value.upper() == "AZ":
            return AgentType.AZ
        supported_types = ", ".join([e.name for e in AgentType])
        message = f"{value} not a supported solver type. Supported types are {supported_types}."
        raise ValueError(message)

    def create_player(self) -> Solver:
        match self:
            case AgentType.HUMAN:
                raise ValueError("todo")
            case AgentType.MINIMAX:
                raise ValueError("todo")
            case AgentType.MCTS:
                from .mcts import MctsSolver

                return MctsSolver()
            case AgentType.AZ:
                raise ValueError("todo")
            case _:
                raise ValueError(f"Unsupported agent type: {self}")

    def create_agent(self, game: Game[TState]) -> Callable[[Observation, Configuration], int]:
        player: Solver = self.create_player()

        def get_best_move(obs: Observation, config: Configuration) -> int:
            state = game.from_kaggle(obs, config)
            move = player.solve(state)
            return move

        return get_best_move
