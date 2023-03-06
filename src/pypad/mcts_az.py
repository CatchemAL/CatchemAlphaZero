from __future__ import annotations

from copy import copy
from math import sqrt
from typing import Generic, List, TypeVar

import torch

from .neural_net import ResNet
from .state import ConnectX, State, TicTacToe

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
            return 0
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


class MctsSolver:
    def __init__(self, neural_net: ResNet) -> None:
        self.neural_net = neural_net

    @torch.no_grad()
    def solve(self, root_state: State[TMove], num_mcts_sims: int = 1_000) -> TMove:
        root: Node[TMove] = Node(root_state, None, None)

        for _ in range(num_mcts_sims):
            node = root
            state = copy(root_state)

            # === Selection ===
            while node.has_children:
                node = node.select_child()
                state.play_move(node.move)

            if legal_moves := list(state.legal_moves()):
                planes = state.to_numpy()
                tensor = torch.tensor(planes).unsqueeze(0)
                raw_policy, value = self.neural_net(tensor)
                raw_policy = torch.softmax(raw_policy, axis=1).squeeze().cpu().numpy()

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
                value: float = value.item()

            else:
                value = 1 if state.is_won() else 0

            # === Backpropagate ===
            while node:
                node.update(value)
                node = node.parent
                value *= -1

        return max(root.children, key=lambda c: c.visit_count).move


import numpy as np
from kaggle_environments import make


def agent_ttt_mcts(obs, config):
    print(config)
    grid = np.asarray(obs.board).reshape(3, 3)
    state = TicTacToe.from_grid(grid)
    mcts = MctsSolver()
    move = mcts.solve(state, 1_000)
    return move


def tictactoe() -> None:
    moves = [0, 5]
    state = TicTacToe.create(moves)
    state.position

    env = make("tictactoe", debug=True)
    env.run([agent_ttt_mcts, agent_ttt_mcts])
    env.render(mode="ipython")


def mcts_az() -> None:
    import numpy as np

    tictactoe = TicTacToe.create()

    print("Starting...")
    shape = 3, 3
    num_resnet_layers = 4
    num_features = 64
    neural_net = ResNet(shape, num_res_blocks=num_resnet_layers, num_features=num_features)
    mcts = MctsSolver(neural_net)
    move = mcts.solve(tictactoe)
    print(f"Done and move is {move}.")
