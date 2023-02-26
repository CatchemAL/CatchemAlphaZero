from dataclasses import dataclass
from typing import List

class MaskUtils:
    def __init__(self, rows: int, cols: int):
        self.rows: int = rows
        self.cols: int = cols
        self.num_slots: int = rows * cols
        self.BOTTOM_ROW: int = self.get_bottom_row()
        self.BOARD_MASK: int = self.get_board_mask()
        
    def get_bottom_row(self) -> int:
        x = 1
        for i in range(self.cols - 1):
            x |= x << self.rows
        return x
    
    def get_col_mask(self, col: int) -> int:
        first_col = (1 << (self.rows - 1)) - 1
        return first_col << (self.rows * col)
        
    def get_board_mask(self) -> int:
        x = self.BOTTOM_ROW << (self.rows - 1)
        return x - self.BOTTOM_ROW


@dataclass
class Board:
    mask_utils: MaskUtils 
    mask: int 
    position: int
    num_moves: int

    def can_play_col(self, col: int) -> None:
        offset = (col + 1) * self.mask_utils.rows - 2
        top_col_bit = 1 << offset
        return (self.mask & top_col_bit) == 0
    
    def play_col(self, col: int) -> None:
        offset = self.mask_utils.rows * col
        col_bit = 1 << offset
        self.play_move(self.mask + col_bit)
    
    def play_move(self, move: int) -> None:
        self.position ^= self.mask
        self.mask |= move
        self.num_moves += 1
   
    def num_slots(self) -> int:
        return (self.mask_utils.rows - 1) * self.mask_utils.cols

    def key(self) -> int:
        return (self.mask + self.mask_utils.BOTTOM_ROW) | self.position

    def is_full(self) -> bool:
        return self.num_moves == self.num_slots()
    
    def is_won(self) -> bool:
        directions = (self.mask_utils.rows - 1, self.mask_utils.rows, self.mask_utils.rows + 1, 1)
        bitboard = self.position ^ self.mask
        for dir in directions:
            bitmask = bitboard & (bitboard >> dir)
            if (bitmask & (bitmask >> 2 * dir)):
                return True

        return False
    
    def possible_moves_mask(self) -> int:
        return (self.mask + self.mask_utils.BOTTOM_ROW) & self.mask_utils.BOARD_MASK;
    
    def win_mask(self) -> int:

        H1 = self.mask_utils.rows
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

        return wm & (self.mask_utils.BOARD_MASK ^ self.mask)
  
    def copy(self) -> 'Board':
        return Board(self.mask_utils, self.mask, self.position, self.num_moves)
    
    @classmethod
    def create(cls, rows: int, cols: int, moves: List[int]) -> 'Board':
        mask = MaskUtils(rows + 1, cols)
        board = cls(mask, 0, 0, 0) 
        for move in moves:
            board.play_col(move - 1)
        return board
  

class Solver:

    move_order = [3, 4, 2, 5, 1, 6, 0]


    def minimax(self, board: Board, alpha: int, beta: int) -> int:
        if board.is_full():
            return 0
        
        win_mask = board.win_mask()
        possible_moves = board.possible_moves_mask()
        if (win_mask & possible_moves):
            return (board.num_slots() - board.num_moves + 1) // 2
        
        max_possible_score = (board.num_slots() - board.num_moves - 1) // 2
        if max_possible_score <= alpha:
            return max_possible_score
        
        beta = min(beta, max_possible_score)
        
        score = -100_000_000
        move_order = Solver.create_order(board.mask_utils.cols)
        
        for col in move_order:
            if not board.can_play_col(col):
                continue  # todo mask col with possible moves and fire that in instead
            
            b = board.copy()
            b.play_col(col)
            score = -self.minimax(b, -beta, -alpha)
            alpha = max(alpha, score)
            if score >= beta:
                return alpha

        return alpha
    
    @staticmethod
    def create_order(cols: int) -> List[int]:
        order: List[int] = [0] * cols

        for i in range(cols):
            if i % 2 == 0:
                order[i] = (cols - i - 1) // 2
            else:
                order[i] = (cols + i) // 2

        return order
    