from __future__ import annotations

from copy import copy
from math import sqrt
from typing import Generic, List, TypeVar


import numpy as np

from ..states import State
from .network import NeuralNetwork

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
    def __init__(self, neural_net: NeuralNetwork) -> None:
        self.neural_net = neural_net

    def solve(self, root_state: State[TMove], num_mcts_sims: int = 10_000) -> TMove:
        root = self.search(root_state, num_mcts_sims)
        max_child = max(root.children, key=lambda c: c.visit_count)
        return max_child.move

    def search(self, root_state: State[TMove], num_mcts_sims: int = 1_000) -> Node:
        root: Node[TMove] = Node(root_state, None, None)

        for _ in range(num_mcts_sims):
            node = root
            state = copy(root_state)

            # === Selection ===
            while node.has_children:
                node = node.select_child()
                state.play_move(node.move)

            if legal_moves := list(state.legal_moves()):
                raw_policy, value = self.neural_net.predict(state)

                policy = raw_policy * 0  # Filter out illegal moves
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
                value = 1.0 if state.is_won() else 0.0

            # === Backpropagate ===
            while node:
                node.update(value)
                node = node.parent
                value *= -1

        return root


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
