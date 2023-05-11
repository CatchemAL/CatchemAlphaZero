from __future__ import annotations

from dataclasses import dataclass

import random
from copy import copy
from math import log, sqrt
from typing import Generic, List

from ..games.state import State, TMove
from . import Solver


class Node(Generic[TMove]):
    __slots__ = ["move", "parent", "played_by", "wins", "visit_count", "children", "unexplored_moves"]

    def __init__(
        self, state: State[TMove], parent: Node[TMove] | None = None, move: TMove | None = None
    ):
        self.move = move
        self.parent = parent
        self.played_by = state.played_by

        self.wins: int = 0
        self.visit_count: int = 0

        self.children: List[Node[TMove]] = []
        self.unexplored_moves: List[TMove] = list(state.legal_moves())

    @property
    def has_legal_moves(self) -> bool:
        return bool(self.children or self.unexplored_moves)

    @property
    def is_leaf_node(self) -> bool:
        return bool(self.unexplored_moves)

    def update(self, outcome: int) -> None:
        self.visit_count += 1
        self.wins += outcome

    def select_child(self) -> "Node[TMove]":
        return max(self.children, key=lambda c: c.ucb())

    def ucb(self) -> float:
        c = sqrt(2)
        exploitation_param = self.wins / self.visit_count
        exploration_param = sqrt(log(self.parent.visit_count) / self.visit_count)
        return exploitation_param + c * exploration_param

    def __repr__(self):
        return f"Node(move={self.move}, W/V={self.wins}/{self.visit_count}, ({len(self.unexplored_moves)} unexplored moves))"


@dataclass
class MctsSolver(Solver):
    num_mcts_sims: int = 1_000

    def solve(self, root_state: State[TMove]) -> TMove:
        root = self.search(root_state)
        max_child = max(root.children, key=lambda c: c.visit_count)
        return max_child.move

    def search(self, root_state: State[TMove]) -> Node:
        root: Node[TMove] = Node(root_state, None, None)

        for _ in range(self.num_mcts_sims):
            node = root
            state = copy(root_state)

            # Selection
            while not node.is_leaf_node and node.has_legal_moves:
                node = node.select_child()
                state.play_move(node.move)

            # Expansion
            if node.unexplored_moves:
                move = random.choice(node.unexplored_moves)
                state.play_move(move)
                child_node = Node(state, parent=node, move=move)
                node.unexplored_moves.remove(move)
                node.children.append(child_node)
                node = child_node

            # Simulation (aka rollout)
            while legal_moves := list(state.legal_moves()):
                move = random.choice(legal_moves)
                state.play_move(move)

            # Backpropagate
            outcome = state.outcome(node.played_by)
            while node:
                node.update(outcome)
                outcome = 1 - outcome
                node = node.parent

        return root
