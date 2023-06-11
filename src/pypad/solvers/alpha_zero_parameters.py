from dataclasses import dataclass, field
from typing import Self

from ..states.state import TemperatureSchedule


@dataclass
class RandomStart:
    num_moves: int
    probability: float


@dataclass
class AZMctsParameters:
    num_mcts_sims: int
    dirichlet_epsilon: float
    dirichlet_alpha: float
    discount_factor: float

    @classmethod
    def defaults(cls, fullname: str) -> Self:
        match fullname:
            case "TicTacToe":
                params = {
                    "num_mcts_sims": 500,
                    "dirichlet_epsilon": 0.25,
                    "dirichlet_alpha": 0.9,
                    "discount_factor": 0.99,
                }
                return cls(**params)

            case "ConnectX_6x7":
                params = {
                    "num_mcts_sims": 600,
                    "dirichlet_epsilon": 0.25,
                    "dirichlet_alpha": 0.6,
                    "discount_factor": 0.98,
                }
                return cls(**params)

            case "Chess":
                params = {
                    "num_mcts_sims": 800,
                    "dirichlet_epsilon": 0.25,
                    "dirichlet_alpha": 0.3,
                    "discount_factor": 0.99,
                }
                return cls(**params)


@dataclass
class AZArenaParameters:
    num_games: int = 400
    num_mcts_sims: int = 60
    required_win_ratio: float = 0.55
    temperature_schedule: TemperatureSchedule = field(
        default_factory=lambda: TemperatureSchedule(4, 1.25)
    )

    @classmethod
    def defaults(cls, fullname: str) -> Self:
        match fullname:
            case "TicTacToe":
                params = {
                    "num_mcts_sims": 500,
                    "dirichlet_epsilon": 0.25,
                    "dirichlet_alpha": 0.9,
                    "discount_factor": 0.99,
                }

                return cls(**params)
            case "ConnectX_6x7":
                params = {
                    "num_mcts_sims": 1600,
                    "dirichlet_epsilon": 0.25,
                    "dirichlet_alpha": 0.6,
                    "discount_factor": 0.98,
                }

                return cls(**params)


@dataclass
class AZTrainingParameters:
    num_generations: int
    num_epochs: int
    games_per_generation: int
    num_parallel: int
    minibatch_size: int
    random_start: RandomStart
    temperature: TemperatureSchedule
    mcts_parameters: AZMctsParameters
    arena_parameters: AZArenaParameters

    @classmethod
    def defaults(cls, fullname: str) -> Self:
        mcts_parameters = AZMctsParameters.defaults(fullname)
        arena_parameters = AZArenaParameters()
        random_start = RandomStart(3, 0.3)

        match fullname:
            case "TicTacToe":
                params = {
                    "num_generations": 50,
                    "num_epochs": 5,
                    "games_per_generation": 100,
                    "num_parallel": 50,
                    "minibatch_size": 128,
                    "random_start": random_start,
                    "temperature": TemperatureSchedule(2, 1.25),
                    "mcts_parameters": mcts_parameters,
                    "arena_parameters": arena_parameters,
                }

                return cls(**params)
            case "ConnectX_6x7":
                params = {
                    "num_generations": 50,
                    "num_epochs": 4,
                    "games_per_generation": 400,
                    "num_parallel": 100,
                    "minibatch_size": 512,
                    "random_start": random_start,
                    "temperature": TemperatureSchedule(10, 1.2),
                    "mcts_parameters": mcts_parameters,
                    "arena_parameters": arena_parameters,
                }

                return cls(**params)

            case "Chess":
                params = {
                    "num_generations": 50,
                    "num_epochs": 4,
                    "games_per_generation": 400,
                    "num_parallel": 100,
                    "minibatch_size": 512,
                    "random_start": random_start,
                    "temperature": TemperatureSchedule(20, 1.2),
                    "mcts_parameters": mcts_parameters,
                    "arena_parameters": arena_parameters,
                }
                return cls(**params)


@dataclass
class AZNetworkParameters:
    num_resnet_blocks: int
    num_features: int
    optimizer_learn_rate: float
    optimizer_weight_decay: float

    @classmethod
    def defaults(cls, fullname: str) -> Self:
        match fullname:
            case "TicTacToe":
                params = {
                    "num_resnet_blocks": 4,
                    "num_features": 64,
                    "optimizer_learn_rate": 0.001,
                    "optimizer_weight_decay": 0.0001,
                }

                return cls(**params)

            case "ConnectX_6x7":
                params = {
                    "num_resnet_blocks": 9,
                    "num_features": 128,
                    "optimizer_learn_rate": 0.001,
                    "optimizer_weight_decay": 0.0001,
                }

                return cls(**params)

            case "Chess":
                params = {
                    "num_resnet_blocks": 19,
                    "num_features": 256,
                    "optimizer_learn_rate": 0.001,
                    "optimizer_weight_decay": 0.0001,
                }

                return cls(**params)
            case _:
                raise NotImplementedError(f"Network defaults for {fullname} not supported.")
