from collections.abc import Generator
from dataclasses import dataclass
from typing import Generator, List, Protocol, Tuple, TypeVar

import numpy as np

from .bitboard_utils import BitboardUtil

TMove = TypeVar("TMove")


class State(Protocol[TMove]):
    @property
    def played_by(self) -> int:
        ...

    @property
    def shape(self) -> Tuple[int, int]:
        ...

    def legal_moves(self) -> Generator[TMove, None, None]:
        ...

    def play_move(self, move: TMove) -> None:
        ...

    def is_won(self) -> bool:
        ...

    def outcome(self, perspective: int, indicator: str = "win-loss") -> float:
        ...

    def to_numpy(self) -> np.ndarray:
        ...

    def __copy__(self) -> "State[TMove]":
        ...


# 3  7 11
# 2  6 10
# 1  5  9
# 0  4  8
MOVES = [1 << 2, 1 << 6, 1 << 10, 1 << 1, 1 << 5, 1 << 9, 1 << 0, 1 << 4, 1 << 8]
POWERS = np.array(MOVES).reshape(3, 3)


@dataclass
class TicTacToe:
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

    def to_numpy(self):
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

    def __copy__(self) -> "TicTacToe":
        return TicTacToe(self.bitboard_util, self.mask, self.position, self.num_moves)

    @classmethod
    def from_grid(cls, grid: np.ndarray) -> "TicTacToe":
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
    def create(cls, moves: List[int] | None = None) -> "TicTacToe":
        mask = BitboardUtil(3 + 1, 3)
        board = cls(mask, 0, 0, 0)
        moves = moves or []
        for move in moves:
            board.play_move(move)
        return board


@dataclass
class ConnectX:
    bitboard_util: BitboardUtil
    mask: int
    position: int
    num_moves: int

    @property
    def rows(self) -> int:
        return self.bitboard_util.rows - 1

    @property
    def cols(self) -> int:
        return self.bitboard_util.cols

    @property
    def shape(self) -> Tuple[int, int]:
        return self.rows, self.cols

    @property
    def num_slots(self) -> int:
        return self.rows * self.cols

    @property
    def played_by(self) -> int:
        is_odd_num_moves = self.num_moves & 1
        return 2 - is_odd_num_moves

    def can_play_col(self, col: int) -> bool:
        offset = (col + 1) * self.bitboard_util.rows - 2
        top_col_bit = 1 << offset
        return (self.mask & top_col_bit) == 0

    def play_col(self, col: int) -> None:
        offset = self.bitboard_util.rows * col
        col_bit = 1 << offset
        self.play_move(self.mask + col_bit)

    def play_move(self, move: int) -> None:
        self.position ^= self.mask
        self.mask |= move
        self.num_moves += 1

    def key(self) -> int:
        return (self.mask + self.bitboard_util.BOTTOM_ROW) | self.position

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
            bitmask = bitboard & (bitboard >> dir)
            if bitmask & (bitmask >> 2 * dir):
                return True

        return False

    def legal_moves(self) -> Generator[int, None, None]:
        if not self.is_won():
            return self.possible_moves()
        return range(0)

    def possible_moves(self) -> Generator[int, None, None]:
        possible_moves_mask = self.possible_moves_mask()
        move_order = self.bitboard_util.move_order()

        for col in move_order:
            col_mask = self.bitboard_util.get_col_mask(col)
            possible_move = possible_moves_mask & col_mask
            if possible_move:
                yield possible_move

    def possible_moves_mask(self) -> int:
        return (self.mask + self.bitboard_util.BOTTOM_ROW) & self.bitboard_util.BOARD_MASK

    def possible_col_moves(self) -> Generator[int, None, None]:
        possible_moves_mask = self.possible_moves_mask()
        move_order = self.bitboard_util.move_order()

        for col in move_order:
            col_mask = self.bitboard_util.get_col_mask(col)
            possible_move = possible_moves_mask & col_mask
            if possible_move > 0:
                yield col

    def win_mask(self) -> int:
        H1 = self.bitboard_util.rows
        posn = self.position

        # Vertical win
        wm = (posn << 1) & (posn << 2) & (posn << 3)

        # Horizontals (_XXXO and OXXX_)
        wm |= (posn << H1) & (posn << 2 * H1) & (posn << 3 * H1)
        wm |= (posn >> H1) & (posn >> 2 * H1) & (posn >> 3 * H1)

        # Horizontals (OXX_XO and OX_XXO)
        wm |= (posn << H1) & (posn << 2 * H1) & (posn >> H1)
        wm |= (posn >> H1) & (posn >> 2 * H1) & (posn << H1)

        # Diagonals _/_
        wm |= (posn << (H1 + 1)) & (posn << 2 * (H1 + 1)) & (posn << 3 * (H1 + 1))
        wm |= (posn << (H1 + 1)) & (posn << 2 * (H1 + 1)) & (posn >> (H1 + 1))
        wm |= (posn << (H1 + 1)) & (posn >> (H1 + 1)) & (posn >> 2 * (H1 + 1))
        wm |= (posn >> (H1 + 1)) & (posn >> 2 * (H1 + 1)) & (posn >> 3 * (H1 + 1))

        # Diagonals _\_
        wm |= (posn >> (H1 - 1)) & (posn >> 2 * (H1 - 1)) & (posn >> 3 * (H1 - 1))
        wm |= (posn >> (H1 - 1)) & (posn >> 2 * (H1 - 1)) & (posn << (H1 - 1))
        wm |= (posn >> (H1 - 1)) & (posn << (H1 - 1)) & (posn << 2 * (H1 - 1))
        wm |= (posn << (H1 - 1)) & (posn << 2 * (H1 - 1)) & (posn << 3 * (H1 - 1))

        return wm & (self.bitboard_util.BOARD_MASK ^ self.mask)

    def __copy__(self) -> "ConnectX":
        return ConnectX(self.bitboard_util, self.mask, self.position, self.num_moves)

    def to_grid(self) -> np.ndarray:
        num_entries = self.num_slots + self.cols
        linear_grid = np.zeros((num_entries,), dtype=np.int8)

        posn = self.position ^ self.mask if self.num_moves & 1 else self.position
        player_1 = posn
        player_2 = posn ^ self.mask

        for i in range(num_entries):
            if player_1 & 1 << i:
                linear_grid[i] = 1
            elif player_2 & 1 << i:
                linear_grid[i] = 2

        shape = self.rows + 1, self.cols
        return np.flipud(linear_grid.reshape(shape).transpose())[1:, :]

    @classmethod
    def from_grid(cls, grid: np.ndarray) -> "ConnectX":
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
    def create(cls, rows: int, cols: int, moves: List[int] | None = None) -> "ConnectX":
        mask = BitboardUtil(rows + 1, cols)
        board = cls(mask, 0, 0, 0)
        moves = moves or []
        for move in moves:
            board.play_col(move - 1)
        return board
