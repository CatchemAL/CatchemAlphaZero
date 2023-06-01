from dataclasses import dataclass

import numpy as np

from ..bitboard_utils import BitboardUtil
from .state import State, Status, TemperatureSchedule

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
        legal_moves = [] if is_ended else self._possible_moves_unchecked()
        return Status(is_in_progress, self.played_by, value, legal_moves)

    def play_move(self, move: int) -> None:
        bitmove = MOVES[move]
        self.position ^= self.mask
        self.mask |= bitmove
        self.num_moves += 1

    def select_move(self, policy: np.ndarray, temperature_schedule: TemperatureSchedule) -> int:
        temperature = temperature_schedule.get_temperature(self.num_moves)
        if temperature < 0.001:
            return np.argmax(policy)

        temperature_policy = policy ** (1 / temperature)
        temperature_policy /= temperature_policy.sum()
        return np.random.choice(len(policy), p=temperature_policy)

    def is_won(self) -> bool:
        rows = self.rows + 1
        directions = (1, rows - 1, rows, rows + 1)
        bitboard = self.position ^ self.mask
        for dir in directions:
            if bitboard & (bitboard >> dir) & (bitboard >> 2 * dir):
                return True

        return False

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

    def html(self, policy: np.ndarray | None = None, is_tiny_repr: bool = False) -> str:
        from ..views.html import TicTacToeHtmlBuilder

        html_printer = TicTacToeHtmlBuilder()

        if is_tiny_repr:
            return html_printer.build_tiny_html(self)

        return html_printer.build_html(self, policy)

    def plot(self):
        from ..views.plot import plot_state

        plot_state(self.to_numpy(), figsize=(3, 2), linewidth=1.5)

    def _possible_moves_unchecked(self) -> list[int]:
        possible_moves_mask = self._possible_bitmoves_mask()
        return [i for i, move in enumerate(MOVES) if possible_moves_mask & move]

    def _possible_bitmoves_mask(self) -> int:
        return self.mask ^ self.bitboard_util.BOARD_MASK

    def _repr_html_(self) -> str:
        return self.html()

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
    def create(cls, moves: str | list[int] | None = None) -> "TicTacToeState":
        mask = BitboardUtil(3 + 1, 3)
        board = cls(mask, 0, 0, 0)
        moves = moves or []

        if isinstance(moves, str):
            move_array = moves.replace(" ", "").split(",")
            moves = [int(move) for move in move_array]

        for move in moves:
            board.play_move(move)
        return board
