import pickle
from copy import deepcopy
from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from numpy.typing import NDArray
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

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
        self,
        resnet: ResNet,
        optimizer: Optimizer,
        game: Game,
        generation: int,
        directory: Path,
    ) -> None:
        self.resnet = resnet
        self.optimizer = optimizer
        self.game = game
        self.directory = directory
        self.generation = generation
        self.device = next(resnet.parameters()).device.type
        self.action_size = game.config().action_size

    @torch.no_grad()
    def predict(self, state: State[TMove]) -> tuple[NDArray[np.float32], float]:
        # Convert to [player, opponent, unplayed]
        planes = state.to_feature()
        tensor = torch.tensor(planes, device=self.device).unsqueeze(0)

        predicted_policy, predicted_outcome = self.resnet(tensor)
        normalized_policy = torch.softmax(predicted_policy, axis=1).squeeze().cpu().numpy()

        return normalized_policy, predicted_outcome.item()

    @torch.no_grad()
    def predict_parallel(self, states: list[State]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        encoded_states = np.array([s.to_feature() for s in states], dtype=np.float32)
        torch_states = torch.tensor(encoded_states, device=self.device)

        predicted_policies, predicted_outcomes = self.resnet(torch_states)
        normalized_policies = torch.softmax(predicted_policies, axis=1).cpu().numpy()

        return normalized_policies, predicted_outcomes.squeeze().cpu().numpy()

    def set_to_eval(self) -> None:
        self.resnet.eval()

    def train(
        self,
        training_set: list[TrainingData],
        num_epochs: int,
        minibatch_size: int,
        log_progress: bool = True,
    ) -> None:
        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir=f"runs/{self.game.fullname}")
        gen = self.generation
        num_points = len(training_set)

        # Convert to numpy arrays for training
        states_np = np.array([d.encoded_state for d in training_set], dtype=np.float32)
        policies_np = np.array([d.policy for d in training_set], dtype=np.float32)
        outcomes_np = np.array([d.outcome for d in training_set], dtype=np.float32).reshape(-1, 1)

        states = torch.tensor(states_np, device=self.device)
        policies = torch.tensor(policies_np, device=self.device)
        outcomes = torch.tensor(outcomes_np, device=self.device)

        data_set = TensorDataset(states, policies, outcomes)
        data_loader = DataLoader(data_set, batch_size=minibatch_size, shuffle=True)

        if log_progress and gen == 0:
            with torch.no_grad():
                batch_states, batch_policies, batch_outcomes = next(iter(data_loader))
                grid = torchvision.utils.make_grid(batch_states)
                writer.add_image("images", grid, 0)
                writer.add_graph(self.resnet, batch_states)

            writer.close()

        # Record how well the previous generation predicts the next set of policies
        if log_progress:
            self.resnet.eval()
            epoch_policy_loss = 0.0
            epoch_outcome_loss = 0.0
            epoch_total_loss = 0.0
            with torch.no_grad():
                for batch_states, batch_policies, batch_outcomes in data_loader:
                    predicted_policies, predicted_outcomes = self.resnet(batch_states)

                    # Compute loss
                    policy_loss = F.cross_entropy(predicted_policies, batch_policies)
                    outcome_loss = F.mse_loss(predicted_outcomes, batch_outcomes)
                    total_loss = policy_loss + outcome_loss

                    # Store key metrics
                    epoch_policy_loss += policy_loss.item()
                    epoch_outcome_loss += outcome_loss.item()
                    epoch_total_loss += total_loss.item()

            # Logging metrics to TensorBoard
            writer.add_scalar(f"Generations/Total Loss", epoch_total_loss / num_points, gen)
            writer.add_scalar(f"Generations/Policy Loss", epoch_policy_loss / num_points, gen)
            writer.add_scalar(f"Generations/Outcome Loss", epoch_outcome_loss / num_points, gen)

        self.resnet.train()
        for epoch in trange(num_epochs, desc=" - Training", leave=False):
            epoch_policy_loss = 0.0
            epoch_outcome_loss = 0.0
            epoch_total_loss = 0.0

            for batch_states, batch_policies, batch_outcomes in data_loader:
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Feed forwards
                predicted_policies, predicted_outcomes = self.resnet(batch_states)

                # Compute loss
                policy_loss = F.cross_entropy(predicted_policies, batch_policies)
                outcome_loss = F.mse_loss(predicted_outcomes, batch_outcomes)
                total_loss = policy_loss + outcome_loss

                # Backwards + optimize
                total_loss.backward()
                self.optimizer.step()

                # Store key metrics
                epoch_policy_loss += policy_loss.item()
                epoch_outcome_loss += outcome_loss.item()
                epoch_total_loss += total_loss.item()

            # Logging metrics to TensorBoard
            if log_progress:
                writer.add_scalar(f"Gen{gen:02d}/Total Loss", epoch_total_loss, epoch + 1)
                writer.add_scalar(f"Gen{gen:02d}/Policy Loss", epoch_policy_loss, epoch + 1)
                writer.add_scalar(f"Gen{gen:02d}/Outcome Loss", epoch_outcome_loss, epoch + 1)

        # Close TensorBoard writer
        writer.close()

    def save_training_data(self, training_set: list[TrainingData]) -> None:
        gen = self.generation
        data_directory = self.directory / "training_data"
        data_file = data_directory / f"{self.game.fullname}_gen{gen:03d}.pkl"

        with open(data_file, "wb") as f:
            pickle.dump(training_set, f)

    def load_training_data(self, gen: int) -> list[TrainingData]:
        data_directory = self.directory / "training_data"
        data_file = data_directory / f"{self.game.fullname}_gen{gen:03d}.pkl"

        with open(data_file, "rb") as f:
            return pickle.load(f)

    def save(self) -> None:
        gen = self.generation
        weights_directory = self.directory / "weights"
        network_weights = self.resnet.state_dict()
        optimizer_state = self.optimizer.state_dict()
        weights_file = weights_directory / f"weights_{self.game.fullname}_gen{gen:03d}.pt"
        optimizer_file = weights_directory / f"optimizer_state_{self.game.fullname}_gen{gen:03d}.pt"
        torch.save(network_weights, weights_file)
        torch.save(optimizer_state, optimizer_file)

    def __deepcopy__(self, memo) -> Self:
        # Deep copy the model and optimizer
        resnet = deepcopy(self.resnet)
        optimizer = type(self.optimizer)(resnet.parameters(), **self.optimizer.defaults)

        resnet.load_state_dict(self.resnet.state_dict())
        optimizer.load_state_dict(self.optimizer.state_dict())

        game = deepcopy(self.game)
        directory = deepcopy(self.directory)

        return PytorchNeuralNetwork(resnet, optimizer, game, self.generation, directory)

    @classmethod
    def create(cls, game: Game, root_directory: str | Path, load_latest: bool | int = True) -> Self:
        root_directory = Path(root_directory)
        directory = root_directory / "weights"
        game_parameters = game.config()
        obs_shape = game_parameters.observation_shape
        action_size = game_parameters.action_size

        device = "cuda" if torch.cuda.is_available() else "cpu"

        net_params = AZNetworkParameters.defaults(game.fullname)
        resnet = ResNet(obs_shape, action_size, net_params.num_resnet_blocks, net_params.num_features)
        resnet = resnet.to(device)

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

        return PytorchNeuralNetwork(resnet, optimizer, game, generation, root_directory)
