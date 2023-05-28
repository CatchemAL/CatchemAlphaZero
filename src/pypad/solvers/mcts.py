from __future__ import annotations

import random
from copy import copy
from dataclasses import dataclass
from math import log, sqrt
from typing import Generic

from ..states import State, Status, TMove, TState
from . import Solver


class Node(Generic[TMove]):
    __slots__ = ["move", "parent", "played_by", "wins", "visit_count", "children", "unexplored_moves"]

    def __init__(
        self, status: Status[TMove], parent: Node[TMove] | None = None, move: TMove | None = None
    ):
        self.move = move
        self.parent = parent
        self.played_by = status.played_by

        self.wins: int = 0
        self.visit_count: int = 0

        self.children: list[Node[TMove]] = []
        self.unexplored_moves: list[TMove] = status.legal_moves

    @property
    def has_legal_moves(self) -> bool:
        return bool(self.children or self.unexplored_moves)

    @property
    def is_leaf_node(self) -> bool:
        return bool(self.unexplored_moves)

    def update(self, outcome: int) -> None:
        self.visit_count += 1
        self.wins += (1 + outcome) / 2

    def select_child(self) -> "Node[TMove]":
        return max(self.children, key=lambda c: c.ucb())

    def ucb(self) -> float:
        c = sqrt(2)
        exploitation_param = self.wins / self.visit_count
        exploration_param = sqrt(log(self.parent.visit_count) / self.visit_count)
        return exploitation_param + c * exploration_param

    def html(self, state: TState) -> str:
        from ..views.html import MctsNodeHtmlBuilder

        html_printer = MctsNodeHtmlBuilder()
        return html_printer.build_tiny_html(self, state)

    def render(self, state: State[TMove]):
        from ..graph import visualize_tree

        return visualize_tree(state, self)

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
        root: Node[TMove] = Node(root_state.status(), None, None)

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
                child_node = Node(state.status(), parent=node, move=move)
                node.unexplored_moves.remove(move)
                node.children.append(child_node)
                node = child_node

            # Simulation (aka rollout)
            while (status := state.status()).is_in_progress:
                move = random.choice(status.legal_moves)
                state.play_move(move)

            # Backpropagate
            while node:
                outcome = status.outcome(node.played_by)
                node.update(outcome)
                node = node.parent

        return root
