from dataclasses import dataclass
from typing import Generator, List, Tuple

import numpy as np

from ..bitboard_utils import BitboardUtil
from ..kaggle_types import Configuration, Observation
from .state import State, StateFactory, StateView


@dataclass
class ConnectX(State[int]):
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


class ConnectXFactory(StateFactory[ConnectX]):
    def load_initial_state(self, initial_position: str) -> ConnectX:
        return ConnectX.create(7, 6, initial_position)

    def from_kaggle(self, obs: Observation, config: Configuration) -> ConnectX:
        grid = np.asarray(obs.board).reshape(config.rows, config.columns)
        state = ConnectX.from_grid(grid)
        return state


class ConnectXView(StateView[ConnectX]):
    def display(self, state: ConnectX) -> None:
        grid = state.to_grid()
        print(grid)


def mcts() -> None:
    import numpy as np

    grid = np.array(
        [
            [0, 1, 1, 2, 2, 2, 0],
            [0, 1, 2, 1, 1, 1, 0],
            [0, 2, 1, 1, 2, 2, 1],
            [0, 2, 2, 2, 1, 1, 2],
            [0, 1, 2, 1, 2, 2, 1],
            [2, 1, 2, 1, 1, 2, 1],
        ]
    )

    ROWS, COLS = 6, 7
    connect = ConnectX.create(ROWS, COLS)
    connect = ConnectX.from_grid(grid)

    print("Starting...")
    mcts = MctsSolver()
    move = mcts.solve(connect)
    print(f"Done and move is {move}.")
