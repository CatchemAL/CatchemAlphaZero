from dataclasses import dataclass
import numpy as np

UNKNOWN = 0
EXCLUDE = 1
INCLUDE = 2


@dataclass
class Board:
    grid: np.ndarray
    mask: np.ndarray
    row_sums: np.ndarray
    col_sums: np.ndarray


class SumpleteSolver:
    def solve(self, board: Board) -> None:
        is_updating = True
        while is_updating and not board.is_solved():
            is_updating = False

            for line in board.unsolved_lines():
                knapsack = Knapsack(line)
                for index in line.unknowns():
                    if knapsack.must_include(index):
                        grid[line.full_index] = INCLUDE
                        knapsack[index] = INCLUDE
                        is_updating = True
                    elif knapsack.must_exclude(index):
                        grid[line.full_index] = EXCLUDE
                        knapsack[index] = EXCLUDE
                        is_updating = True
