from enum import Enum

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

    def create_player(self, game) -> Solver:
        match self:
            case AgentType.HUMAN:
                raise ValueError("todo")
            case AgentType.MINIMAX:
                raise ValueError("todo")
            case AgentType.MCTS:
                from .mcts import MctsSolver

                return MctsSolver()
            case AgentType.AZ:
                from ..alpha_zero import AlphaZero, PytorchNeuralNetwork

                neural_net = PytorchNeuralNetwork.create(game, ".")
                alpha_zero = AlphaZero(neural_net)
                return alpha_zero.as_solver(num_mcts_sims=100)
            case _:
                raise ValueError(f"Unsupported agent type: {self}")
