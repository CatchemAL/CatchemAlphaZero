from dataclasses import dataclass
import numpy as np
from typing import List
from collections.abc import Generator

class BitboardUtil:
    def __init__(self, rows: int, cols: int):
        self.rows: int = rows
        self.cols: int = cols
        self.num_slots: int = rows * cols
        self.BOTTOM_ROW: int = self.get_bottom_row()
        self.BOARD_MASK: int = self.get_board_mask()
        
    def get_bottom_row(self) -> int:
        x = 1
        for _ in range(self.cols - 1):
            x |= x << self.rows
        return x
    
    def get_col_mask(self, col: int) -> int:
        first_col = (1 << (self.rows - 1)) - 1
        return first_col << (self.rows * col)
        
    def get_board_mask(self) -> int:
        x = self.BOTTOM_ROW << (self.rows - 1)
        return x - self.BOTTOM_ROW
    
    def move_order(self) -> List[int]:
        order: List[int] = [0] * self.cols

        for i in range(self.cols):
            if i % 2 == 0:
                order[i] = (self.cols - i - 1) // 2
            else:
                order[i] = (self.cols + i) // 2

        return order


@dataclass
class Board:
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

    def can_play_col(self, col: int) -> None:
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
   
    def num_slots(self) -> int:
        return self.rows * self.cols

    def key(self) -> int:
        return (self.mask + self.bitboard_util.BOTTOM_ROW) | self.position

    def is_full(self) -> bool:
        return self.num_moves == self.num_slots()
    
    def is_won(self) -> bool:
        directions = (self.bitboard_util.rows - 1, self.bitboard_util.rows, self.bitboard_util.rows + 1, 1)
        bitboard = self.position ^ self.mask
        for dir in directions:
            bitmask = bitboard & (bitboard >> dir)
            if (bitmask & (bitmask >> 2 * dir)):
                return True

        return False
    
    def possible_moves_mask(self) -> int:
        return (self.mask + self.bitboard_util.BOTTOM_ROW) & self.bitboard_util.BOARD_MASK
    
    def possible_moves(self) -> Generator[int, None, None]:
        possible_moves_mask = self.possible_moves_mask()
        move_order = self.bitboard_util.move_order()
        
        for col in move_order:
            col_mask = self.bitboard_util.get_col_mask(col)
            possible_move = possible_moves_mask & col_mask
            if possible_move > 0:
                yield possible_move
    
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
        wm |= (posn << (H1 + 1)) & (posn << 2 * (H1 + 1)) & (posn >>     (H1 + 1))
        wm |= (posn << (H1 + 1)) & (posn >>     (H1 + 1)) & (posn >> 2 * (H1 + 1))
        wm |= (posn >> (H1 + 1)) & (posn >> 2 * (H1 + 1)) & (posn >> 3 * (H1 + 1))

        # Diagonals _\_
        wm |= (posn >> (H1 - 1)) & (posn >> 2 * (H1 - 1)) & (posn >> 3 * (H1 - 1))
        wm |= (posn >> (H1 - 1)) & (posn >> 2 * (H1 - 1)) & (posn <<     (H1 - 1))
        wm |= (posn >> (H1 - 1)) & (posn <<     (H1 - 1)) & (posn << 2 * (H1 - 1))
        wm |= (posn << (H1 - 1)) & (posn << 2 * (H1 - 1)) & (posn << 3 * (H1 - 1))

        return wm & (self.bitboard_util.BOARD_MASK ^ self.mask);

    def copy(self) -> 'Board':
        return Board(self.bitboard_util, self.mask, self.position, self.num_moves)
    
    @classmethod
    def create(cls, rows: int, cols: int, moves: List[int]) -> 'Board':
        mask = BitboardUtil(rows + 1, cols)
        board = cls(mask, 0, 0, 0) 
        for move in moves:
            board.play_col(move - 1)
        return board
    
    @classmethod
    def from_grid(cls, grid) -> 'Board':
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
        board.num_moves = np.sum(mask)
        return board