from dataclasses import dataclass
from typing import Self


@dataclass
class AZTrainingParameters:
    num_generations: int
    num_epochs: int
    games_per_generation: int
    num_mcts_sims: int
    minibatch_size: int
    temperature: float
    dirichlet_epsilon: float
    dirichlet_alpha: float

    @classmethod
    def defaults(cls, fullname: str) -> Self:
        match fullname:
            case "TicTacToe":
                params = {
                    "num_generations": 50,
                    "num_epochs": 10,
                    "games_per_generation": 100,
                    "num_mcts_sims": 500,
                    "minibatch_size": 64,
                    "temperature": 1.25,
                    "dirichlet_epsilon": 0.25,
                    "dirichlet_alpha": 0.9,
                }

                return cls(**params)
            case "ConnectX_6x7":
                params = {
                    "num_generations": 50,
                    "num_epochs": 10,
                    "games_per_generation": 500,
                    "num_mcts_sims": 800,
                    "minibatch_size": 64,
                    "temperature": 1.25,
                    "dirichlet_epsilon": 0.25,
                    "dirichlet_alpha": 0.9,
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
            case _:
                raise NotImplementedError(f"Network defaults for {fullname} not supported.")
