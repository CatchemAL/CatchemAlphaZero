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
class KnapsackNumberLine:
    def __init__(self, numbers: np.ndarray, mask: np.ndarray) -> None:
        self.numbers: np.ndarray = numbers
        self.mask: np.ndarray = mask

        self._number_line, self._accessibles = self._build_accessible_numbers()

    def can_reach(self, target: int) -> bool:
        return self._accessibles[self._number_line == target]

    def _build_accessible_numbers(self) -> Tuple[np.ndarray, np.ndarray]:
        lower_bound, upper_bound = self.get_bounds()
        number_line = np.arange(lower_bound, upper_bound + 1)
        accessibles = number_line == 0

        for num, m in zip(self.numbers, self.mask):
            if m == UNKNOWN:
                shifted = KnapsackNumberLine.shift(accessibles, num)
                accessibles |= shifted

        include_shift = np.sum(self.numbers[self.mask == INCLUDE])
        accessibles = KnapsackNumberLine.shift(accessibles, include_shift)

        return (number_line, accessibles)

    def get_bounds(self) -> np.ndarray:
        lower_bound = np.sum(self.numbers[self.numbers < 0])
        upper_bound = np.sum(self.numbers[self.numbers > 0])
        return lower_bound, upper_bound

    @staticmethod
    def shift(x: np.ndarray, n: int) -> np.ndarray:
        if n == 0:
            return x[:]

        if n > 0:
            return np.pad(x, (n, 0))[:-n]

        return np.pad(x, (0, -n))[-n:]


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

    def without(self, index: int) -> KnapsackNumberLine:
        pass

    def include(self, index: int) -> None:
        self.mask[index] = INCLUDE

    def exclude(self, index: int) -> None:
        self.mask[index] = EXCLUDE


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
