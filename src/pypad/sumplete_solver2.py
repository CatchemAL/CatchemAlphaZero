from enum import Enum
from dataclasses import dataclass
from typing import Generator, List, Tuple
import numpy as np

UNKNOWN = 0
EXCLUDE = 1
INCLUDE = 2


class VectorType(Enum):
    ROW = 1
    COLUMN = 2


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
class BoardVector:
    numbers: np.ndarray
    mask: np.ndarray
    target: float
    vector_type: VectorType
    index: int

    def unknowns(self) -> Generator[int, None, None]:
        for i, m in enumerate(self.mask):
            if m == UNKNOWN:
                yield i

    def full_index(self, index: int) -> Tuple[int, int]:
        if self.vector_type == VectorType.COLUMN:
            return index, self.index
        if self.vector_type == VectorType.ROW:
            return self.index, index
        raise ValueError(f"vector {self.vector_type} type unknown")

    def without(self, index: int) -> Knapsack:
        pass


@dataclass
class Board:
    grid: np.ndarray
    mask: np.ndarray
    row_sums: np.ndarray
    col_sums: np.ndarray

    def unsolved_vectors(self) -> Generator[BoardVector, None, None]:
        yield from self.unsolved_rows()
        yield from self.unsolved_cols()

    def unsolved_rows(self) -> Generator[BoardVector, None, None]:
        for i in range(self.grid.shape[0]):
            row_mask = self.mask[i, :]
            if np.any(row_mask == UNKNOWN):
                yield BoardVector(self.grid[i, :], self.mask[i, :], self.row_sums[i])

    def unsolved_cols(self) -> Generator[BoardVector, None, None]:
        for j in range(self.grid.shape[1]):
            col_mask = self.mask[:, j]
            if np.any(col_mask == UNKNOWN):
                yield BoardVector(self.grid[:, j], self.mask[:, j], self.col_sums[j])

    def is_solved(self) -> bool:
        return True


class SumpleteSolver:
    def solve(self, board: Board) -> None:
        is_updating = True
        while is_updating and not board.is_solved():
            is_updating = False

            for vector in board.unsolved_vectors():
                for index in vector.unknowns():
                    number_line = vector.without(index)
                    i, j = vector.full_index(index)
                    if not number_line.can_reach(vector.target):
                        vector.include(index)
                        board[i, j] = INCLUDE
                        is_updating = True
                    elif not number_line.can_reach(vector.target - vector.numbers[index]):
                        vector.exclude(index)
                        board[i, j] = EXCLUDE
                        is_updating = True
