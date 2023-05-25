from dataclasses import dataclass

from typing import Self


@dataclass
class AZTrainingParameters:
    num_generations: int
    num_epochs: int
    games_per_generation: int
    num_mcts_sims: int
    minibatch_size: int

    @classmethod
    def defaults(cls, fullname: str) -> Self:
        match fullname:
            case "TicTacToe":
                params = {
                    "num_generations": 5,
                    "num_epochs": 4,
                    "games_per_generation": 1_000,
                    "num_mcts_sims": 200,
                    "minibatch_size": 64,
                }

                return cls(**params)


@dataclass
class AZNetworkParameters:
    num_resnet_blocks: int
    num_features: int
    optimizer_learn_rate: float

    @classmethod
    def defaults(cls, fullname: str) -> Self:
        match fullname:
            case "TicTacToe":
                params = {
                    "num_resnet_blocks": 4,
                    "num_features": 64,
                    "optimizer_learn_rate": 0.001,
                }

                return cls(**params)
