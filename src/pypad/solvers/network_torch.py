from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, TensorDataset

from ..games import Game
from ..states import State, TMove
from .alpha_zero_parameters import AZNetworkParameters
from .network import TrainingData

KERNEL_SIZE = 3
PADDING = (KERNEL_SIZE - 1) // 2


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


class PytorchNeuralNetwork:
    def __init__(
        self, resnet: ResNet, optimizer: Optimizer, game: Game, generation: int, directory: Path
    ) -> None:
        self.resnet = resnet
        self.optimizer = optimizer
        self.game = game
        self.directory = directory
        self.generation = generation
        self.action_size = game.config().action_size

    @torch.no_grad()
    def predict(self, state: State[TMove]) -> tuple[NDArray[np.float32], float]:
        # Convert to [player, opponent, unplayed]
        planes = state.to_numpy()
        tensor = torch.tensor(planes).unsqueeze(0)

        predicted_policy, predicted_outcome = self.resnet(tensor)
        normalized_policy = torch.softmax(predicted_policy, axis=1).squeeze().cpu().numpy()

        return normalized_policy, predicted_outcome.item()

    @torch.no_grad()
    def predict_parallel(self, state: list[State]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        batch_states = torch.from_numpy(np.array([s.to_numpy() for s in state]))

        predicted_policies, predicted_outcomes = self.resnet(batch_states)
        normalized_policies = torch.softmax(predicted_policies, axis=1).cpu().numpy()

        return normalized_policies, predicted_outcomes.squeeze().cpu().numpy()

    def set_to_eval(self) -> None:
        self.resnet.eval()

    def train(self, training_set: list[TrainingData], minibatch_size: int) -> None:
        self.resnet.train()

        states = torch.from_numpy(np.array([d.encoded_state for d in training_set]))
        policies = torch.from_numpy(np.array([d.policy for d in training_set]))
        outcomes = torch.tensor([d.outcome for d in training_set], dtype=torch.float32).reshape(-1, 1)

        data_set = TensorDataset(states, policies, outcomes)
        data_loader = DataLoader(data_set, batch_size=minibatch_size, shuffle=True)

        for batch_states, batch_policies, batch_outcomes in data_loader:
            predicted_policies, predicted_outcomes = self.resnet(batch_states)

            policy_loss = F.cross_entropy(predicted_policies, batch_policies)
            outcome_loss = F.mse_loss(predicted_outcomes, batch_outcomes)
            total_loss = policy_loss + outcome_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    def save(self) -> None:
        gen = self.generation
        network_weights = self.resnet.state_dict()
        optimizer_state = self.optimizer.state_dict()
        weights_file = self.directory / f"weights_{self.game.fullname}_gen{gen:03d}.pt"
        optimizer_file = self.directory / f"optimizer_state_{self.game.fullname}_gen{gen:03d}.pt"
        torch.save(network_weights, weights_file)
        torch.save(optimizer_state, optimizer_file)

    @classmethod
    def create(cls, game: Game, directory: str | Path, load_latest: bool | int = True) -> Self:
        directory = Path(directory)
        game_parameters = game.config()
        obs_shape = game_parameters.observation_shape
        action_size = game_parameters.action_size

        net_params = AZNetworkParameters.defaults(game.fullname)
        resnet = ResNet(obs_shape, action_size, net_params.num_resnet_blocks, net_params.num_features)

        learning_rate = net_params.optimizer_learn_rate
        l2_regularization = net_params.optimizer_weight_decay
        optimizer = Adam(resnet.parameters(), lr=learning_rate, weight_decay=l2_regularization)

        weights_regex = f"weights_{game.fullname}_gen*.pt"
        optimizer_regex = f"optimizer_state_{game.fullname}_gen*.pt"
        weights_files = [file_path for file_path in directory.glob(weights_regex)]
        optim_files = [file_path for file_path in directory.glob(optimizer_regex)]

        generation = 0
        if load_latest:
            if not isinstance(load_latest, bool):
                weights_files = [f for f in weights_files if int(f.stem.split("gen")[1]) == load_latest]
                optim_files = [f for f in optim_files if int(f.stem.split("gen")[1]) == load_latest]

            if weights_files:
                latest_weights_file = max(weights_files, key=lambda f: int(f.stem.split("gen")[1]))
                resnet.load_state_dict(torch.load(latest_weights_file))
                generation = int(latest_weights_file.stem.split("gen")[1])

            if optim_files:
                latest_optimiser_file = max(optim_files, key=lambda f: int(f.stem.split("gen")[1]))
                optimizer.load_state_dict(torch.load(latest_optimiser_file))

        return PytorchNeuralNetwork(resnet, optimizer, game, generation, directory)
