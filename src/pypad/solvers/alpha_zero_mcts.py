from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from math import sqrt
from typing import Generic, Self

import numpy as np

from ..states import State, TMove
from ..states.state import Policy
from .network import NeuralNetwork


@dataclass
class AlphaZeroMcts:
    neural_net: NeuralNetwork
    num_mcts_sims: int
    dirichlet_epsilon: float
    dirichlet_alpha: float
    discount_factor: float

    def policy(self, state: State[TMove]) -> Policy[TMove]:
        root_node = self.search(state)
        root_q = 1 - root_node.q_value
        value = root_q * 2 - 1

        moves = [node.move for node in root_node.children]
        priors = np.array([node.visit_count for node in root_node.children], dtype=np.float32)
        priors /= priors.sum()

        policy_shape = self.neural_net.game.config().action_size
        encoded_policy = np.zeros(policy_shape)
        for move, prior in zip(moves, priors):
            loc = state.policy_loc(move)
            encoded_policy[loc] = prior

        return Policy(moves, priors, encoded_policy, value)

    def policies(self, states: list[State[TMove]]) -> list[Policy]:
        root_nodes = self.search_parallel(states)

        policies: list[Policy] = []
        policy_shape = self.neural_net.game.config().action_size

        for state, root_node in zip(states, root_nodes):
            root_q = 1 - root_node.q_value
            value = root_q * 2 - 1

            moves = [node.move for node in root_node.children]
            priors = np.array([node.visit_count for node in root_node.children], dtype=np.float32)
            priors /= priors.sum()

            encoded_policy = np.zeros(policy_shape)
            for move, prior in zip(moves, priors):
                loc = state.policy_loc(move)
                encoded_policy[loc] = prior

            policy = Policy(moves, priors, encoded_policy, value)
            policies.append(policy)

        return policies

    def search(self, root_state: State[TMove]) -> Node[TMove]:
        root: Node[TMove] = Node(root_state.played_by)

        for i in range(self.num_mcts_sims):
            node = root
            state = copy(root_state)

            # === Selection ===
            while node.has_children:
                node = node.select_child()
                state.set_move(node.move)

            status = state.status()
            if status.is_in_progress:
                raw_policy, value = self.neural_net.predict(state)

                if i == 0:
                    ε = self.dirichlet_epsilon
                    alpha = self.dirichlet_alpha
                    noise = np.random.dirichlet([alpha] * len(raw_policy))
                    raw_policy = (1 - ε) * raw_policy + ε * noise

                # Get the priors
                priors = np.zeros(len(status.legal_moves))
                for i, move in enumerate(status.legal_moves):
                    loc = state.policy_loc(move)
                    priors[i] = raw_policy[loc]
                priors /= np.sum(priors)

                # === Expansion ===
                for move, prior in zip(status.legal_moves, priors):
                    child_state = state.play_move(move)
                    child_node = Node(child_state.played_by, parent=node, move=move, prior=prior)
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

    def search_parallel(self, root_states: list[State[TMove]]) -> list[Node]:
        action_size = self.neural_net.action_size

        roots = [Node(root_state.played_by) for root_state in root_states]

        for i in range(self.num_mcts_sims):
            nodes = [root for root in roots]
            states: list[State[TMove]] = [copy(root_state) for root_state in root_states]

            # === Selection ===
            for j in range(len(nodes)):
                while nodes[j].has_children:
                    nodes[j] = nodes[j].select_child()
                    states[j].set_move(nodes[j].move)

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
                        child_state = states[idx].play_move(move)
                        prior = policies[i, move]
                        child_node = Node(child_state.played_by, nodes[idx], move, prior)
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


@dataclass(slots=True)
class Node(Generic[TMove]):
    played_by: int
    parent: Node[TMove] | None = None
    move: TMove | None = None
    prior: np.float32 = 1.0

    value_sum: float = field(default_factory=float, init=False)
    visit_count: int = field(default_factory=int, init=False)
    children: list[Node[TMove]] = field(default_factory=list, init=False)

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

    def select_child(self) -> Self:
        return max(self.children, key=lambda c: c.ucb())

    def ucb(self) -> float:
        c = 2  # todo AlphaZero sets to.... 2?
        exploration_param = sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + c * self.prior * exploration_param

    def __repr__(self):
        if self.parent:
            return f"Node(move={self.move}, Q={self.q_value:.1%}, prior={self.prior:.2%}, visit_count={self.visit_count}, UCB={self.ucb():.3})"
        return f"Node(move={self.move}, Q={self.q_value:.1%}, prior={self.prior:.2%}, visit_count={self.visit_count})"
