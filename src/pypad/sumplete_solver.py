import numpy as np
from dataclasses import dataclass
from typing import Tuple

UNKNOWN: int = 0
EXCLUDE: int = 1
INCLUDE: int = 2


@dataclass
class Board:
    grid: np.ndarray
    row_totals: np.ndarray
    col_totals: np.ndarray


@dataclass
class SumpleteSolver:
    def solve(self, board: Board) -> None:
        mask = board.grid * UNKNOWN

    def get_bounds(grid: np.ndarray, axis: int) -> Tuple[int, int]:
        """
        Returns a tuple containing the minimum and maximum row or column-wise sums of a numpy array

        Parameters:
            grid (numpy.ndarray): A matrix of integers
            axis (int, optional): Axis along which the sum is to be computed. Default is 0 for columns.

        Returns:
            tuple: A tuple containing the minimum and maximum sums along the specified axis.
        """
        pos_mask = grid > 0
        neg_mask = grid < 0

        neg_sum_cols = np.min(np.sum(grid * neg_mask, axis=0))
        pos_sum_cols = np.max(np.sum(grid * pos_mask, axis=0))
        neg_sum_rows = np.min(np.sum(grid * neg_mask, axis=1))
        pos_sum_rows = np.max(np.sum(grid * pos_mask, axis=1))

        lower_bound = min(neg_sum_cols, neg_sum_rows)
        upper_bound = max(pos_sum_cols, pos_sum_rows)

        return (lower_bound, upper_bound)
