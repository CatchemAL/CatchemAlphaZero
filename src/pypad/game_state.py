from typing import List, TypeVar, Protocol, Generic, Generator
from copy import copy
from dataclasses import dataclass
from math import sqrt, log
import random

TMove = TypeVar("TMove")


class State(Protocol[TMove]):
    def get_legal_moves(self) -> Generator[TMove, None, None]:
        ...

    def play_move(self, move: TMove) -> None:
        ...

    def __copy__(self) -> "State[TMove]":
        ...


class Node(Generic[TMove]):
    __slots__ = ["move", "parent", "children", "wins", "visits", "untried_moves", "player_just_moved"]

    def __init__(self, state: State[TMove], parent: "Node" = None, move: TMove = None):
        self.move: TMove = move
        self.parent: "Node[TMove]" = parent
        self.children: List["Node[TMove]"] = []

        self.unexplored_moves: List[TMove] = list(state.get_legal_moves())
        self.player_just_moved: int = state.player_just_moved

        self.wins: int = 0
        self.visit_count: int = 0

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


class ConnectXState:
    def get_legal_moves(self) -> List[int]:
        pass

    def play_move(self, move: int) -> None:
        pass


class MctsSolver:
    def solve(self, root_state: State[TMove], num_mcts_sims: int = 1_000) -> TMove:
        root: Node[TMove] = Node(root_state)

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
                child = Node(state, parent=node, move=move)
                node.unexplored_moves.remove(move)
                node.children.append(child)

            # Simulation (aka rollout)
            while legal_moves := list(state.get_legal_moves()):
                move = random.choice(legal_moves)
                state.play_move(move)

            # Backpropagate
            outcome = state.get_outcome()
            while node:
                node.update(outcome)
                outcome *= -1
                node = node.parent

        return max(root.children, key=lambda c: c.visit_count).move


if __name__ == "__main__":
    connect = ConnectXState()
    mcts = MctsSolver()
    result = mcts.solve(connect)
