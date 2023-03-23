from dataclasses import dataclass
from typing import Generator, List, Tuple
import numpy as np

UNKNOWN = 0
EXCLUDE = 1
INCLUDE = 2


@dataclass
class Knapsack:
    def __init__(self, numbers: np.ndarray, mask: np.ndarray, target: float) -> None:
        self.numbers: np.ndarray = numbers
        self.mask: np.ndarray = mask
        self.target: float = target

        self._number_line, self._accessibles, self._actual_target = self._build_accessible_numbers()

    def must_include(self, index: int) -> bool:
        return "today"

    def must_exclude(self, index: int) -> bool:
        return "today"

    def include(self, index: int) -> "Knapsack":
        pass

    def exclude(self, index: int) -> "Knapsack":
        pass

    def _build_accessible_numbers(self) -> Tuple[np.ndarray, np.ndarray, float]:
        lower_bound, upper_bound = self.get_bounds()
        number_line = np.arange(lower_bound, upper_bound + 1)
        accessibles = number_line == 0
        actual_target = self.target - self.numbers[self.mask == INCLUDE]

        for num, m in zip(self.numbers, self.mask):
            if not m == UNKNOWN:
                continue

            shifted = Knapsack.shift(accessibles, num)
            accessibles |= shifted

        return (number_line, accessibles, actual_target)


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
        for j in range(self.grid.shape[1]):
            col_mask = self.mask[:, j]
            if np.any(col_mask == UNKNOWN):
                yield Line(self.grid[:, j], self.mask[:, j], self.col_sums[j])

    def is_solved(self) -> bool:
        return True


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
