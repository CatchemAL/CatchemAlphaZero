from dataclasses import dataclass
from typing import Generator, List, Tuple

import numpy as np

from ..bitboard_utils import BitboardUtil
from .state import State

# 3  7 11
# 2  6 10
# 1  5  9
# 0  4  8

MOVES = [1 << 2, 1 << 6, 1 << 10, 1 << 1, 1 << 5, 1 << 9, 1 << 0, 1 << 4, 1 << 8]
POWERS = np.array(MOVES).reshape(3, 3)


@dataclass
class TicTacToeState(State[int]):
    bitboard_util: BitboardUtil
    mask: int
    position: int
    num_moves: int

    @property
    def rows(self) -> int:
        return 3

    @property
    def cols(self) -> int:
        return 3

    @property
    def shape(self) -> Tuple[int, int]:
        return 3, 3

    @property
    def num_slots(self) -> int:
        return 9

    @property
    def played_by(self) -> int:
        is_odd_num_moves = self.num_moves & 1
        return 2 - is_odd_num_moves

    def play_move(self, move: int) -> None:
        bitmove = MOVES[move]
        self.position ^= self.mask
        self.mask |= bitmove
        self.num_moves += 1

    def is_full(self) -> bool:
        return self.num_moves == self.num_slots

    def outcome(self, perspective: int, indicator: str = "win-loss") -> float:
        score = self._outcome(perspective)
        if indicator == "win-loss":
            return (1 + np.sign(score)) / 2

        return score

    def _outcome(self, perspective: int) -> int:
        if self.is_full and not self.is_won():
            return 0

        score = (self.num_slots - self.num_moves + 2) // 2
        is_odd_num_moves = self.num_moves & 1
        is_odd_perspective = perspective & 1

        return score if is_odd_num_moves == is_odd_perspective else -score

    def is_won(self) -> bool:
        rows = self.rows + 1
        directions = (1, rows - 1, rows, rows + 1)
        bitboard = self.position ^ self.mask
        for dir in directions:
            if bitboard & (bitboard >> dir) & (bitboard >> 2 * dir):
                return True

        return False

    def legal_moves(self) -> Generator[int, None, None]:
        if not self.is_won():
            return self.possible_moves()
        return range(0)

    def possible_moves(self) -> Generator[int, None, None]:
        possible_moves_mask = self.possible_moves_mask()

        for i, move in enumerate(MOVES):
            if possible_moves_mask & move:
                yield i

    def possible_moves_mask(self) -> int:
        return self.mask ^ self.bitboard_util.BOARD_MASK

    def to_numpy(self) -> np.ndarray:
        player_to_move = self.position
        opponent_of_player_to_move = self.position ^ self.mask
        r = np.sign(player_to_move & POWERS, dtype=np.float32)
        g = np.sign(opponent_of_player_to_move & POWERS, dtype=np.float32)
        b = 1 - r - g
        return np.stack((r, g, b))

    def to_grid(self) -> np.ndarray:
        posn = self.position ^ self.mask if self.num_moves & 1 else self.position
        player_1 = posn
        player_2 = posn ^ self.mask
        r = np.sign(player_1 & POWERS)
        g = np.sign(player_2 & POWERS)
        return np.asarray(r + 2 * g, dtype=np.int8)

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        grid = self.to_grid()
        r = (grid == 1).astype(np.float32)
        g = (grid == 2).astype(np.float32)
        b = (grid == 0).astype(np.float32)
        planes = [r, g, b]
        stacked = np.stack(planes)

        _, ax = plt.subplots(figsize=(3, 2))
        plt.imshow(stacked.transpose(1, 2, 0))
        ax.set_xticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Player 1 = Red\nPlayer 2 = Green", loc="left", fontsize=8, fontname="Monospace")
        plt.show()

    def __copy__(self) -> "TicTacToeState":
        return TicTacToeState(self.bitboard_util, self.mask, self.position, self.num_moves)

    @classmethod
    def from_grid(cls, grid: np.ndarray) -> "TicTacToeState":
        rows, cols = grid.shape
        padded_grid = np.vstack((np.zeros(cols), grid))

        indices = np.flipud(np.arange((rows + 1) * cols).reshape((cols, rows + 1)).transpose())
        binary_vals = 2 ** indices.astype(np.int64)

        mask = (padded_grid > 0).astype(np.int64)
        num_moves = np.sum(mask)
        mark = 1 + num_moves % 2

        posn = (padded_grid == mark).astype(np.int64)
        mask_utils = BitboardUtil(rows + 1, cols)
        board = cls(mask_utils, 0, 0, 0)
        mask_val = np.sum(mask * binary_vals)
        posn_val = np.sum(posn * binary_vals)
        board.mask = mask_val
        board.position = posn_val
        board.num_moves = int(np.sum(mask))
        return board

    @classmethod
    def create(cls, moves: List[int] | None = None) -> "TicTacToeState":
        mask = BitboardUtil(3 + 1, 3)
        board = cls(mask, 0, 0, 0)
        moves = moves or []
        for move in moves:
            board.play_move(move)
        return board
