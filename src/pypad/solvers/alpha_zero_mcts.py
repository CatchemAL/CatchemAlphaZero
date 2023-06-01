from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from math import sqrt
from typing import Generic

import numpy as np

from ..states import State, TMove
from .network import NeuralNetwork


class Node(Generic[TMove]):
    __slots__ = ["move", "parent", "played_by", "value_sum", "visit_count", "children", "prior"]

    def __init__(
        self,
        state: State[TMove],
        parent: Node[TMove] | None = None,
        move: TMove | None = None,
        prior: np.float32 = 1.0,
    ):
        self.move = move
        self.parent = parent
        self.played_by = state.played_by
        self.prior = prior

        self.value_sum: float = 0.0
        self.visit_count: int = 0

        self.children: list[Node[TMove]] = []

    @property
    def has_children(self) -> bool:
        return bool(self.children)

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            if self.parent is None:
                return 0

            first_play_urgency = 0.44
            q_from_parent = 1 - self.parent.q_value
            estimated_q_value = q_from_parent - first_play_urgency
            return max(estimated_q_value, 0)

        return (1 + self.value_sum / self.visit_count) / 2

    def update(self, outcome: float) -> None:
        self.visit_count += 1
        self.value_sum += outcome

    def backpropagate(self, outcome: float, discount_factor: float) -> None:
        self.update(outcome)
        if self.parent:
            self.parent.backpropagate(-outcome * discount_factor, discount_factor)

    def select_child(self) -> "Node[TMove]":
        return max(self.children, key=lambda c: c.ucb())

    def ucb(self) -> float:
        c = 2  # todo AlphaZero sets to.... 2?
        exploration_param = sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + c * self.prior * exploration_param

    def __repr__(self):
        if self.parent:
            return f"Node(move={self.move}, Q={self.q_value:.1%}, prior={self.prior:.2%}, visit_count={self.visit_count}, UCB={self.ucb():.3})"
        return f"Node(move={self.move}, Q={self.q_value:.1%}, prior={self.prior:.2%}, visit_count={self.visit_count})"


@dataclass
class AlphaZeroMctsSolver:
    neural_net: NeuralNetwork
    num_mcts_sims: int
    dirichlet_epsilon: float
    dirichlet_alpha: float
    discount_factor: float

    def solve(self, root_state: State[TMove]) -> TMove:
        root = self.search(root_state)
        max_child = max(root.children, key=lambda c: c.visit_count)
        return max_child.move

    def policy(self, state: State[TMove], is_raw_policy: bool = False) -> tuple[np.ndarray, float]:
        if is_raw_policy:
            raw_policy, value = self.neural_net.predict(state)
            return raw_policy, value

        root = self.search(state)
        root_q = 1 - root.q_value
        value = root_q * 2 - 1

        policy = np.zeros(self.neural_net.action_size, dtype=np.float32)
        visit_counts = [node.visit_count for node in root.children]
        moves = [node.move for node in root.children]
        policy[moves] = visit_counts
        policy /= policy.sum()

        return policy, value

    def policies(self, states: list[State[TMove]]) -> tuple[np.ndarray, np.ndarray]:
        roots = self.search_parallel(states)

        root_q = np.array([1 - root.q_value for root in roots])
        values = root_q * 2 - 1

        shape = len(states), self.neural_net.action_size
        policy = np.zeros(shape, dtype=np.float32)
        for i, root in enumerate(roots):
            visit_counts = [node.visit_count for node in root.children]
            moves = [node.move for node in root.children]
            policy[i, moves] = visit_counts

        policy /= policy.sum(axis=1)[:, np.newaxis]
        return policy, values

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
            node.backpropagate(value, self.discount_factor)

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
                node.backpropagate(values[i], self.discount_factor)

        return roots
