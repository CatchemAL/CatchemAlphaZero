import torch
import torch.nn as nn
import torch.optim as optim

from ..states import State, TMove
from dataclasses import dataclass

from numpy.typing import NDArray
import numpy as np

NUM_CHANNELS = 3  # Player mask, Opponent mask, Possible moves mask
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
        self, shape: tuple[int, int], action_size: int, num_res_blocks: int = 4, num_features: int = 64
    ) -> None:
        super().__init__()

        rows, cols = shape

        # We start with a convolutional layer
        self.start_block = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, num_features, kernel_size=KERNEL_SIZE, padding=PADDING),
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
            # nn.Softmax()
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
