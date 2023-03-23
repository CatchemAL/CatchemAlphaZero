from dataclasses import dataclass
from typing import Generator, List, Tuple
import numpy as np

UNKNOWN = 0
EXCLUDE = 1
INCLUDE = 2


class Knapsack:
    def must_include(self) -> bool:
        return "today"

    def must_exclude(self) -> bool:
        return "today"

    def include(self, i: int, j: int) -> "Knapsack":
        pass

    def exclude(self, i: int, j: int) -> "Knapsack":
        pass


@dataclass
class Line:
    numbers: np.ndarray
    mask: np.ndarray
    target: float

    def unknowns(self) -> Generator[int, None, None]:
        for i, m in enumerate(self.mask):
            if m == UNKNOWN:
                yield i


@dataclass
class Board:
    grid: np.ndarray
    mask: np.ndarray
    row_sums: np.ndarray
    col_sums: np.ndarray

    def unsolved_lines(self) -> Generator[Line, None, None]:
        yield from self.unsolved_rows()
        yield from self.unsolved_cols()

    def unsolved_rows(self) -> Generator[Line, None, None]:
        for i in range(self.grid.shape[0]):
            row_mask = self.mask[i, :]
            if np.any(row_mask == UNKNOWN):
                yield Line(self.grid[i, :], self.mask[i, :], self.row_sums[i])

    def unsolved_cols(self) -> Generator[Line, None, None]:
        for i in range(self.grid.shape[1]):
            col_mask = self.mask[:, i]
            if np.any(col_mask == UNKNOWN):
                yield Line(self.grid[:, i], self.mask[:, i], self.col_sums[i])


class SumpleteSolver:
    def solve(self, board: Board) -> None:
        is_updating = True
        while is_updating and not board.is_solved():
            is_updating = False

            for line in board.unsolved_lines():
                knapsack = Knapsack(line)
                for index in line.unknowns():
                    if knapsack.must_include(index):
                        knapsack = knapsack.include(line.full_index)
                        board[line.full_index] = INCLUDE
                        is_updating = True
                    elif knapsack.must_exclude(index):
                        knapsack = knapsack.exclude(line.full_index)
                        board[line.full_index] = EXCLUDE
                        is_updating = True
