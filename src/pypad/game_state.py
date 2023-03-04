from __future__ import annotations

import random
from copy import copy
from math import log, sqrt
from typing import Generator, Generic, List, Protocol, TypeVar

from .connectx import Board

TMove = TypeVar("TMove")


class State(Protocol[TMove]):
    @property
    def played_by(self) -> int:
        ...

    def possible_moves(self) -> Generator[TMove, None, None]:
        ...

    def play_move(self, move: TMove) -> None:
        ...

    def is_won(self) -> bool:
        ...

    def outcome(self, perspective: int, indicator: str = "win-loss") -> float:
        ...

    def __copy__(self) -> "State[TMove]":
        ...


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
        self.unexplored_moves: List[TMove] = list(state.possible_moves())

    @property
    def has_legal_moves(self) -> bool:
        return bool(self.children or self.unexplored_moves)

    @property
    def is_leaf_node(self) -> bool:
        return any(self.unexplored_moves)

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


class MctsSolver:
    def solve(self, root_state: State[TMove], num_mcts_sims: int = 1_000) -> TMove:
        root: Node[TMove] = Node(root_state, None, None)

        for _ in range(num_mcts_sims):
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
            while not state.is_won() and (legal_moves := list(state.possible_moves())):
                move = random.choice(legal_moves)
                state.play_move(move)

            # Backpropagate
            outcome = state.outcome(node.played_by)
            while node:
                node.update(outcome)
                outcome *= -1
                node = node.parent

        return max(root.children, key=lambda c: c.visit_count).move


def mcts() -> None:
    ROWS, COLS = 6, 7
    connect = Board.create(ROWS, COLS)
    print("Starting...")
    mcts = MctsSolver()
    move = mcts.solve(connect)
    print(f"Done and move is {move}.")
