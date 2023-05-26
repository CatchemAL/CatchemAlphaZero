from __future__ import annotations
from dataclasses import dataclass
import random
from copy import copy
from math import sqrt
from typing import Generic, List, Self, TypeVar

import numpy as np
from tqdm import tqdm, trange

from .alpha_zero_parameters import AZTrainingParameters

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


@dataclass
class AlphaZeroMctsSolver:
    neural_net: NeuralNetwork
    num_mcts_sims: int = 1_000

    def solve(self, root_state: State[TMove]) -> TMove:
        root = self.search(root_state)
        max_child = max(root.children, key=lambda c: c.visit_count)
        return max_child.move

    def policy(self, state: State[TMove]) -> np.ndarray:
        root = self.search(state)

        policy = np.zeros(self.neural_net.action_size, dtype=np.float32)
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


@dataclass
class AlphaZero:
    neural_net: NeuralNetwork

    def self_play(
        self, num_mcts_sims: int, initial_state: str | list[int] | None = None
    ) -> list[TrainingData]:
        state: State = self.neural_net.game.initial_state(initial_state)
        status = state.status()
        mcts = AlphaZeroMctsSolver(self.neural_net, num_mcts_sims)

        self.neural_net.set_to_eval()
        move_history: list[tuple[np.ndarray, np.ndarray, int]] = []

        temperature = 1
        while status.is_in_progress:
            encoded_state = state.to_numpy()
            policy = mcts.policy(state)
            temperature_policy = policy ** (1 / temperature)
            move = np.random.choice(len(policy), p=temperature_policy)
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

    def train(self, training_set: list[TrainingData], minibatch_size: int) -> None:
        random.shuffle(training_set)

        for idx in range(0, len(training_set), minibatch_size):
            minibatch = training_set[idx : idx + minibatch_size]
            self.neural_net.train(minibatch)

    def self_learn(
        self, training_params: AZTrainingParameters, initial_state: str | list[int] | None = None
    ) -> None:
        for _ in trange(training_params.num_generations, desc="Generations"):
            training_set: list[TrainingData] = []

            for _ in trange(training_params.games_per_generation, desc="- Self-play", leave=False):
                game_training_set = self.self_play(training_params.num_mcts_sims, initial_state)
                training_set += game_training_set

            self.neural_net.set_to_train()
            for _ in trange(training_params.num_epochs, desc=" - Training", leave=False):
                self.train(training_set, training_params.minibatch_size)

            self.neural_net.generation += 1
            self.neural_net.save()
