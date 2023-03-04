from __future__ import annotations

from typing import List, TypeVar, Protocol, Generic, Generator, Optional
from copy import copy
from .connectx import Board
from math import sqrt, log
import random

TMove = TypeVar("TMove")


class State(Protocol[TMove]):
    def possible_moves(self) -> Generator[TMove, None, None]:
        ...

    def play_move(self, move: TMove) -> None:
        ...

    def get_outcome(self, player: int) -> int:
        ...

    def __copy__(self) -> "State[TMove]":
        ...


class Node(Generic[TMove]):
    __slots__ = ["move", "parent", "wins", "visit_count", "children", "unexplored_moves"]

    def __init__(
        self, state: State[TMove], parent: Node[TMove] | None = None, move: TMove | None = None
    ):
        self.move = move
        self.parent = parent

        self.wins: int = 0
        self.visit_count: int = 0

        self.children: List[Node[TMove]] = []
        self.unexplored_moves: List[TMove] = list(state.possible_moves())

    @property
    def has_legal_moves(self) -> bool:
        return bool(self.children or self.unexplored_moves)

    @property
    def is_leaf_node(self) -> bool:
        return bool(self.unexplored_moves or self.is_terminal_state)

    def update(self) -> None:
        pass

    def select_child(self) -> "Node[TMove]":
        return max(self.children, key=lambda c: c.ucb())

    def ucb(self) -> float:
        c = sqrt(2)
        exploitation_param = self.wins / self.visits
        exploration_param = sqrt(log(self.parent.visits) / self.visits)
        return exploitation_param + c * exploration_param


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

            # Simulation (aka rollout)
            while legal_moves := list(state.possible_moves()):
                move = random.choice(legal_moves)
                state.play_move(move)

            # Backpropagate
            outcome = state.get_outcome()
            while node:
                node.update(outcome)
                outcome *= -1
                node = node.parent

        return max(root.children, key=lambda c: c.visit_count).move


def mcts() -> None:
    ROWS, COLS = 6, 7
    connect = Board.create(ROWS, COLS)
    mcts = MctsSolver()
    result = mcts.solve(connect)
