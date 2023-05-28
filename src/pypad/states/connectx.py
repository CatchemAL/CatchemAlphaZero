from dataclasses import dataclass
from typing import Generator, Self

import numpy as np

from ..bitboard_utils import BitboardUtil
from .state import State, Status


@dataclass
class ConnectXState(State[int]):
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
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    @property
    def num_slots(self) -> int:
        return self.rows * self.cols

    @property
    def is_full(self) -> bool:
        return self.num_moves == self.num_slots

    @property
    def played_by(self) -> int:
        is_odd_num_moves = self.num_moves & 1
        return 2 - is_odd_num_moves

    def status(self) -> Status[int]:
        is_won = self.is_won()
        is_ended = is_won or self.is_full

        is_in_progress = not is_ended
        value = 1 if is_won else 0
        legal_moves = [] if is_ended else list(self._possible_moves_unchecked())
        return Status(is_in_progress, self.played_by, value, legal_moves)

    def can_play_move(self, col: int) -> bool:
        offset = (col + 1) * self.bitboard_util.rows - 2
        top_col_bit = 1 << offset
        return (self.mask & top_col_bit) == 0

    def play_move(self, col: int) -> None:
        offset = self.bitboard_util.rows * col
        col_bit = 1 << offset
        self.play_bitmove(self.mask + col_bit)

    def play_bitmove(self, move: int) -> None:
        self.position ^= self.mask
        self.mask |= move
        self.num_moves += 1

    def select_move(self, policy: np.ndarray, temperature: float) -> int:
        temperature_policy = policy ** (1 / temperature)
        temperature_policy /= temperature_policy.sum()
        return np.random.choice(len(policy), p=temperature_policy)

    def key(self) -> int:
        return (self.mask + self.bitboard_util.BOTTOM_ROW) | self.position

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

    def legal_bitmoves(self) -> Generator[int, None, None]:
        if self.is_won() or self.is_full:
            return range(0)

        return self._possible_bitmoves_unchecked()

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

    def to_grid(self) -> np.ndarray:
        rows, cols = self.shape
        sequence = np.arange((rows + 1) * cols, dtype=object)
        indices = np.flipud(sequence.reshape((cols, rows + 1)).T)
        powers = 2 ** indices[1:]

        posn = self.position ^ self.mask if self.num_moves & 1 else self.position
        player_1 = posn
        player_2 = posn ^ self.mask
        r = np.sign(player_1 & powers).astype(np.int8)
        g = np.sign(player_2 & powers).astype(np.int8)
        return np.asarray(r + 2 * g)

    def to_numpy(self) -> np.ndarray:
        rows, cols = self.shape
        sequence = np.arange((rows + 1) * cols, dtype=object)
        indices = np.flipud(sequence.reshape((cols, rows + 1)).T)
        powers = 2 ** indices[1:]

        player_1 = self.position
        player_2 = self.position ^ self.mask
        r = np.sign(player_1 & powers).astype(np.float32)
        g = np.sign(player_2 & powers).astype(np.float32)
        b = 1 - r - g
        return np.stack((r, g, b))

    def html(self, policy: np.ndarray | None = None, is_tiny_repr: bool = False) -> str:
        from ..views.html import ConnectXHtmlBuilder

        html_printer = ConnectXHtmlBuilder()

        if is_tiny_repr:
            return html_printer.build_tiny_html(self)

        return html_printer.build_html(self, policy)

    def plot(self):
        from ..views.plot import plot_state

        plot_state(self.to_numpy(), figsize=(4, 4))

    def _possible_moves_unchecked(self) -> Generator[int, None, None]:
        possible_moves_mask = self._possible_bitmoves_mask()
        move_order = self.bitboard_util.move_order()

        for col in move_order:
            col_mask = self.bitboard_util.get_col_mask(col)
            possible_move = possible_moves_mask & col_mask
            if possible_move:
                yield col

    def _possible_bitmoves_unchecked(self) -> Generator[int, None, None]:
        possible_moves_mask = self._possible_bitmoves_mask()
        move_order = self.bitboard_util.move_order()

        for col in move_order:
            col_mask = self.bitboard_util.get_col_mask(col)
            possible_move = possible_moves_mask & col_mask
            if possible_move:
                yield possible_move

    def _possible_bitmoves_mask(self) -> int:
        return (self.mask + self.bitboard_util.BOTTOM_ROW) & self.bitboard_util.BOARD_MASK

    def _repr_html_(self) -> str:
        return self.html()

    def __copy__(self) -> "ConnectXState":
        return ConnectXState(self.bitboard_util, self.mask, self.position, self.num_moves)

    @classmethod
    def from_grid(cls, grid: np.ndarray) -> Self:
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
    def create(cls, rows: int, cols: int, moves: str | list[int] | None = None) -> Self:
        util = BitboardUtil(rows + 1, cols)
        board = cls(util, 0, 0, 0)
        moves = moves or []

        if isinstance(moves, str):
            move_array = moves.replace(" ", "").split(",")
            moves = [int(move) for move in move_array]

        for move in moves:
            board.play_move(move - 1)
        return board
