from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from math import sqrt
from typing import Generic, TypeVar

import numpy as np
from tqdm import trange

from ..games import Game
from ..states import State
from .alpha_zero_parameters import AZTrainingParameters
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

        self.children: list[Node[TMove]] = []

    @property
    def has_children(self) -> bool:
        return bool(self.children)

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return (1 + self.value_sum / self.visit_count) / 2

    def update(self, outcome: float) -> None:
        self.visit_count += 1
        self.value_sum += outcome

    def backpropagate(self, outcome: float) -> None:
        self.update(outcome)
        if self.parent:
            self.parent.backpropagate(-outcome)

    def select_child(self) -> "Node[TMove]":
        return max(self.children, key=lambda c: c.ucb())

    def ucb(self) -> float:
        c = 2  # todo AlphaZero sets to.... 2?
        exploration_param = sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + c * self.prior * exploration_param

    def __repr__(self):
        return f"Node(move={self.move}, Q={self.q_value:.3}, prior={self.prior:.2%}, visit_count={self.visit_count}, UCB={self.ucb():.3}))"


@dataclass
class AlphaZeroMctsSolver:
    neural_net: NeuralNetwork
    num_mcts_sims: int
    dirichlet_epsilon: float
    dirichlet_alpha: float

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

    def policies(self, states: list[State[TMove]]) -> np.ndarray:
        roots = self.search_parallel(states)

        shape = len(states), self.neural_net.action_size
        policy = np.zeros(shape, dtype=np.float32)
        for i, root in enumerate(roots):
            visit_counts = [node.visit_count for node in root.children]
            moves = [node.move for node in root.children]
            policy[i, moves] = visit_counts

        policy /= policy.sum(axis=1)[:, np.newaxis]
        return policy

    def search(self, root_state: State[TMove]) -> Node:
        root: Node[TMove] = Node(root_state, None, None)

        for i in range(self.num_mcts_sims):
            node = root
            state = copy(root_state)

            # === Selection ===
            while node.has_children:
                node = node.select_child()
                state.play_move(node.move)

            status = state.status()
            if status.is_in_progress:
                raw_policy, value = self.neural_net.predict(state)

                if i == 0:
                    ε = self.dirichlet_epsilon
                    alpha = self.dirichlet_alpha
                    noise = np.random.dirichlet([alpha] * len(raw_policy))
                    raw_policy = (1 - ε) * raw_policy + ε * noise

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
                # Here, the AlphaZero paper completely replaces the traditional rollout phase with
                # a value estimation from the neural net.
                # Negate because the net gives an estimation from player whose turn it is next,
                # rather than the player who has just moved
                value *= -1
            else:
                value = status.value

            # === Backpropagate ===
            node.backpropagate(value)

        return root

    def search_parallel(self, root_states: State[TMove]) -> list[Node]:
        action_size = self.neural_net.action_size

        roots: list[Node[TMove]] = [Node(root_state, None, None) for root_state in root_states]

        for i in range(self.num_mcts_sims):
            nodes = [root for root in roots]
            states: list[State[TMove]] = [copy(root_state) for root_state in root_states]

            # === Selection ===
            for j in range(len(nodes)):
                while nodes[j].has_children:
                    nodes[j] = nodes[j].select_child()
                    states[j].play_move(nodes[j].move)

            statuses = [state.status() for state in states]
            in_progress_idxs = [i for i, status in enumerate(statuses) if status.is_in_progress]
            finished_idxs = [i for i, status in enumerate(statuses) if not status.is_in_progress]
            num_in_progress = len(in_progress_idxs)
            values = np.zeros(len(statuses), dtype=np.float32)

            if num_in_progress > 0:
                in_progress_states = [states[idx] for idx in in_progress_idxs]
                raw_policies, predicted_values = self.neural_net.predict_parallel(in_progress_states)

                if i == 0:
                    ε = self.dirichlet_epsilon
                    alpha = self.dirichlet_alpha
                    noise = np.random.dirichlet([alpha] * action_size, (num_in_progress,))
                    policies = (1 - ε) * raw_policies + ε * noise
                else:
                    policies = raw_policies.copy()

                # Filter out illegal moves
                legal_moves_mask = np.zeros((num_in_progress, action_size), dtype=bool)
                for i, idx in enumerate(in_progress_idxs):
                    legal_moves = statuses[idx].legal_moves
                    legal_moves_mask[i, legal_moves] = True
                policies[~legal_moves_mask] = 0
                policies /= policies.sum(axis=1)[:, np.newaxis]

                # === Expansion ===
                for i, idx in enumerate(in_progress_idxs):
                    legal_moves = statuses[idx].legal_moves
                    for move in legal_moves:
                        child_state = copy(states[idx])
                        child_state.play_move(move)
                        prior = policies[i, move]
                        child_node = Node(child_state, parent=nodes[idx], move=move, prior=prior)
                        nodes[idx].children.append(child_node)

                # === Simulation ===
                # Here, the AlphaZero paper completely replaces the traditional rollout phase with
                # a value estimation from the neural net.
                # Negate because the net gives an estimation from player whose turn it is next,
                # rather than the player who has just moved
                values[in_progress_idxs] = -1 * predicted_values

            if finished_idxs:
                values[finished_idxs] = [statuses[idx].value for idx in finished_idxs]

            # === Backpropagate ===
            for i, node in enumerate(nodes):
                node.backpropagate(values[i])

        return roots


@dataclass
class RecordedPolicy:
    state_before: State
    policy: np.ndarray
    move: int
    state_after: State


@dataclass
class ParallelGame:
    idx: int
    initial_state: State
    policy_history: list[RecordedPolicy] = field(default_factory=list)

    @property
    def latest_state(self) -> State:
        if self.policy_history:
            return self.policy_history[-1].state_after
        return self.initial_state


@dataclass
class AlphaZero:
    neural_net: NeuralNetwork

    @property
    def game(self) -> Game:
        return self.neural_net.game

    def self_play(
        self, training_params: AZTrainingParameters, initial_state: str | list[int] | None = None
    ) -> list[TrainingData]:
        mcts = AlphaZeroMctsSolver(
            self.neural_net,
            training_params.num_mcts_sims,
            training_params.dirichlet_epsilon,
            training_params.dirichlet_alpha,
        )

        state: State = self.game.initial_state(initial_state)
        status = state.status()

        self.neural_net.set_to_eval()
        policy_history: list[RecordedPolicy] = []

        while status.is_in_progress:
            state_before = state
            policy = mcts.policy(state_before)
            move = state_before.select_move(policy, training_params.temperature)
            state = copy(state_before)
            state.play_move(move)
            recorded_policy = RecordedPolicy(state_before, policy, move, state)
            policy_history.append(recorded_policy)
            status = state.status()

        training_set: list[TrainingData] = []
        for ph in policy_history:
            encoded_state = ph.state_before.to_numpy()
            perspective = ph.state_after.played_by
            outcome = status.outcome(perspective)
            training_data = TrainingData(encoded_state, ph.policy, outcome)
            training_set.append(training_data)

        return training_set

    def self_play_parallel(
        self, training_params: AZTrainingParameters, initial_state: str | list[int] | None = None
    ) -> list[TrainingData]:
        init_state: State = self.game.initial_state(initial_state)
        initial_status = init_state.status()
        in_progress = initial_status.is_in_progress
        if not in_progress:
            return []

        mcts = AlphaZeroMctsSolver(
            self.neural_net,
            training_params.num_mcts_sims,
            training_params.dirichlet_epsilon,
            training_params.dirichlet_alpha,
        )

        num_parallel = training_params.num_games_in_parallel
        parallel_games = [ParallelGame(i, init_state) for i in range(num_parallel)]
        in_progress_games = [pg for pg in parallel_games]

        self.neural_net.set_to_eval()

        while in_progress_games:
            states = [pg.latest_state for pg in in_progress_games]
            policies = mcts.policies(states)

            for i, pg in enumerate(in_progress_games):
                state_before = pg.latest_state
                policy = policies[i]
                move = state_before.select_move(policy, training_params.temperature)
                state = copy(state_before)
                state.play_move(move)
                recorded_policy = RecordedPolicy(state_before, policy, move, state)
                pg.policy_history.append(recorded_policy)

            in_progress_games = [g for g in in_progress_games if g.latest_state.status().is_in_progress]

        training_set: list[TrainingData] = []
        for pg in parallel_games:
            terminal_status = pg.latest_state.status()
            policy_history = pg.policy_history
            for ph in policy_history:
                encoded_state = ph.state_before.to_numpy()
                perspective = ph.state_after.played_by
                outcome = terminal_status.outcome(perspective)
                training_data = TrainingData(encoded_state, ph.policy, outcome)
                training_set.append(training_data)

        return training_set

    def self_learn(
        self, training_params: AZTrainingParameters, initial_state: str | list[int] | None = None
    ) -> None:
        for _ in trange(training_params.num_generations, desc="Generations"):
            training_set: list[TrainingData] = []

            num_rounds = training_params.games_per_generation // training_params.num_games_in_parallel
            for _ in trange(num_rounds, desc="- Self-play", leave=False):
                # training_set += self.self_play(training_params, initial_state)
                training_set += self.self_play_parallel(training_params, initial_state)

            extended_training_set = self.exploit_symmetries(training_set)

            for _ in trange(training_params.num_epochs, desc=" - Training", leave=False):
                self.neural_net.train(extended_training_set, training_params.minibatch_size)

            self.neural_net.set_to_eval()
            self.neural_net.generation += 1
            self.neural_net.save()

    def exploit_symmetries(self, training_set: list[TrainingData]) -> list[TrainingData]:
        def all_sets(data: TrainingData) -> list[TrainingData]:
            symmetries = self.game.symmetries(data.encoded_state, data.policy)
            return (TrainingData(state, pol, data.outcome) for state, pol in symmetries)

        return [sym_data for training_data in training_set for sym_data in all_sets(training_data)]
