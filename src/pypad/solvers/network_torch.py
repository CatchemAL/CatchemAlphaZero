from dataclasses import dataclass


from .alpha_zero_parameters import AZNetworkParameters

from torch.optim import Adam
from typing import Self

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.optim import Optimizer

from .network import TrainingData
from ..states import State, TMove
from ..games import Game

KERNEL_SIZE = 3
PADDING = (KERNEL_SIZE - 1) // 2

num_res_blocks_by_game = {
    "TicTacToe": 4,
    "Connect4": 9,
}

num_features_by_game = {
    "TicTacToe": 64,
    "Connect4": 128,
}


class ResNet(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        action_size: int,
        num_res_blocks: int = 4,
        num_features: int = 64,
    ) -> None:
        super().__init__()

        rows, cols, num_planes = observation_shape

        # We start with a convolutional layer
        self.start_block = nn.Sequential(
            nn.Conv2d(num_planes, num_features, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
        )

        # This feeds into 'num_res_blocks' residual layers
        self.res_blocks = nn.ModuleList([ResNetBlock(num_features) for _ in range(num_res_blocks)])

        # The value head derives a value, v, for the current position. v ∈ [-1, +1]
        self.value_head = nn.Sequential(
            nn.Conv2d(num_features, 3, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(3 * rows * cols, 1),
            nn.Tanh(),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_features, 32, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * rows * cols, action_size),
        )

    def forward(self, x):
        output = self.start_block(x)
        for res_block in self.res_blocks:
            output = res_block(output)

        policy = self.policy_head(output)
        value = self.value_head(output)

        return policy, value


class ResNetBlock(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.BatchNorm2d(num_features),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.layers(x)
        output += x
        return self.relu(output)


@dataclass
class PytorchNeuralNetwork:
    resnet: ResNet
    optimizer: Optimizer

    @torch.no_grad()
    def predict(self, state: State[TMove]) -> tuple[NDArray[np.float32], float]:
        # Convert to [player, opponent, unplayed]
        planes = state.to_numpy()
        tensor = torch.tensor(planes).unsqueeze(0)

        # Fire the state through the net...
        raw_policy, value = self.resnet(tensor)

        # Surely we can put this in the network architecture instead of outside?
        raw_policy = torch.softmax(raw_policy, axis=1).squeeze().cpu().numpy()

        value: float = value.item()

        return raw_policy, value

    def set_to_eval(self) -> None:
        self.resnet.eval()

    def set_to_train(self) -> None:
        self.resnet.train()

    def train(self, training_set: list[TrainingData]) -> None:
        states = torch.tensor(np.array([d.encoded_state for d in training_set]), dtype=torch.float32)
        policies = torch.tensor(np.array([d.policy for d in training_set]), dtype=torch.float32)
        outcomes = torch.tensor(
            np.array([d.outcome for d in training_set]).reshape(-1, 1), dtype=torch.float32
        )

        predicted_policies, predicted_outcomes = self.resnet(states)

        policy_loss = F.cross_entropy(predicted_policies, policies)
        outcome_loss = F.mse_loss(predicted_outcomes, outcomes)
        total_loss = policy_loss + outcome_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def save(self, generation: int) -> None:
        network_weights = self.resnet.state_dict()
        optimizer_state = self.optimizer.state_dict()
        torch.save(network_weights, f"weights\model_weights_gen{generation:03d}.pt")
        torch.save(optimizer_state, f"weights\optimizer_state_gen{generation:03d}.pt")

    @classmethod
    def create(cls, game: Game, directory: str, load_latest: bool | int = True) -> Self:
        game_parameters = game.config()
        obs_shape = game_parameters.observation_shape
        action_size = game_parameters.action_size

        net_params = AZNetworkParameters.defaults(game.fullname)
        resnet = ResNet(obs_shape, action_size, net_params.num_resnet_blocks, net_params.num_features)
        optimizer = Adam(resnet.parameters(), lr=net_params.optimizer_learn_rate)

        return PytorchNeuralNetwork(resnet, optimizer)
