from __future__ import annotations

import random
from copy import copy
from math import sqrt
from typing import Generic, List, TypeVar

import numpy as np
from tqdm import tqdm, trange

from ..games import Game
from ..states import State
from .network import NeuralNetwork, TrainingData

TMove = TypeVar("TMove")


class Node(Generic[TMove]):
    __slots__ = ["move", "parent", "played_by", "value_sum", "visit_count", "children", "prior"]

    def __init__(
        self,
        state: State[TMove],
        parent: Node[TMove] | None = None,
        move: TMove | None = None,
        prior: np.float32 = 1,
    ):
        self.move = move
        self.parent = parent
        self.played_by = state.played_by
        self.prior = prior

        self.value_sum: int = 0
        self.visit_count: int = 0

        self.children: List[Node[TMove]] = []

    @property
    def has_legal_moves(self) -> bool:
        return bool(self.children or self.unexplored_moves)

    @property
    def has_children(self) -> bool:
        return bool(self.children)

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return (1 + self.value_sum / self.visit_count) / 2

    def update(self, outcome: int) -> None:
        self.visit_count += 1
        self.value_sum += outcome

    def select_child(self) -> "Node[TMove]":
        return max(self.children, key=lambda c: c.ucb())

    def ucb(self) -> float:
        c = 2  # todo AlphaZero sets to.... 2?
        exploration_param = sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + c * self.prior * exploration_param

    def __repr__(self):
        return f"Node(move={self.move}, Q={self.q_value:.4}, prior={self.prior:.2%}))"


class AlphaZeroMctsSolver:
    def __init__(self, neural_net: NeuralNetwork, num_mcts_sims: int = 1_000) -> None:
        self.neural_net = neural_net
        self.num_mcts_sims = num_mcts_sims

    def solve(self, root_state: State[TMove]) -> TMove:
        root = self.search(root_state)
        max_child = max(root.children, key=lambda c: c.visit_count)
        return max_child.move

    def policy(self, state: State[TMove]) -> np.ndarray:
        root = self.search(state)

        policy = np.zeros(state.action_size, dtype=np.float32)
        visit_counts = [node.visit_count for node in root.children]
        moves = [node.move for node in root.children]
        policy[moves] = visit_counts
        policy /= policy.sum()

        return policy

    def search(self, root_state: State[TMove]) -> Node:
        root: Node[TMove] = Node(root_state, None, None)

        for _ in range(self.num_mcts_sims):
            node = root
            state = copy(root_state)

            # === Selection ===
            while node.has_children:
                node = node.select_child()
                state.play_move(node.move)

            status = state.status()
            if status.is_in_progress:
                raw_policy, value = self.neural_net.predict(state)

                # Filter out illegal moves
                legal_moves = status.legal_moves
                policy = raw_policy * 0
                policy[legal_moves] = raw_policy[legal_moves]
                policy /= np.sum(policy)

                # === Expansion ===
                for move in legal_moves:
                    child_state = copy(state)
                    child_state.play_move(move)
                    prior = policy[move]
                    child_node = Node(child_state, parent=node, move=move, prior=prior)
                    node.children.append(child_node)

                # === Simulation ===
                # Here, the AlphaZero paper completely replaces the traditional
                # rollout phase with a value estimation from the neural net.
                ...
            else:
                value = status.value

            # === Backpropagate ===
            while node:
                node.update(value)
                node = node.parent
                value *= -1

        return root


class AlphaZero:
    def __init__(
        self,
        neural_net: NeuralNetwork,
        num_mcts_sims: int = 2_00,
        num_generations: int = 5,
        num_epochs: int = 4,
        games_per_generation: int = 500,
        minibatch_size: int = 64,
    ) -> None:
        self.neural_net = neural_net
        self.mcts = AlphaZeroMctsSolver(self.neural_net, num_mcts_sims)
        self.num_generations = num_generations
        self.num_epochs = num_epochs
        self.games_per_generation = games_per_generation
        self.minibatch_size = minibatch_size

    def self_play(self, game: Game) -> list[TrainingData]:
        state: State = game.initial_state()
        status = state.status()

        move_history: list[tuple[np.ndarray, np.ndarray, int]] = []

        while status.is_in_progress:
            encoded_state = state.to_numpy()
            policy = self.mcts.policy(state)
            move = np.random.choice(state.action_size, p=policy)
            state.play_move(move)
            move_history.append((encoded_state, policy, state.played_by))
            status = state.status()

        training_set: list[TrainingData] = []
        for mh in move_history:
            encoded_state, policy, player_to_move_next = mh
            outcome = status.outcome(player_to_move_next)
            training_data = TrainingData(encoded_state, policy, outcome)
            training_set.append(training_data)

        return training_set

    def train(self, training_set: list[TrainingData]) -> None:
        random.shuffle(training_set)

        for idx in range(0, len(training_set), self.minibatch_size):
            minibatch = training_set[idx : idx + self.minibatch_size]
            self.neural_net.train(minibatch)

    def learn(self, game: Game) -> None:
        outer_progress = tqdm(total=self.num_generations, desc="Generations")
        for generation in range(self.num_generations):
            training_set: list[TrainingData] = []

            self.neural_net.set_to_eval()
            with tqdm(total=self.games_per_generation, desc="- Self-play", leave=False) as inner_bar:
                for _ in range(self.games_per_generation):
                    inner_bar.update(1)
                    game_training_set = self.self_play(game)
                    training_set += game_training_set

            self.neural_net.set_to_train()
            with tqdm(total=self.num_epochs, desc=" - Training", leave=False) as inner_bar:
                for _ in range(self.num_epochs):
                    inner_bar.update(1)
                    self.train(training_set)

            self.neural_net.save(generation)
            outer_progress.update(1)


def mcts_az() -> None:
    import numpy as np

    from ..games import TicTacToe
    from .network import DummyNeuralNetwork

    tictactoe_state = TicTacToe.initial_state()

    print("Starting...")
    shape = 3, 3
    num_resnet_layers = 4
    num_features = 64

    # neural_net = ResNet(shape, num_res_blocks=num_resnet_layers, num_features=num_features)
    neural_net = DummyNeuralNetwork()
    mcts = AlphaZeroMctsSolver(neural_net)
    move = mcts.solve(tictactoe_state)
    print(f"Done and move is {move}.")
