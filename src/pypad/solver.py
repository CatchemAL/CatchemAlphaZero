from .connectx import Board
from copy import copy


class Solver:
    def minimax(self, board: Board, alpha: int, beta: int) -> int:
        if board.is_full():
            return 0

        win_mask = board.win_mask()
        possible_moves = board.possible_moves_mask()
        if win_mask & possible_moves:
            return (board.num_slots() - board.num_moves + 1) // 2

        max_possible_score = (board.num_slots() - board.num_moves - 1) // 2
        if max_possible_score <= alpha:
            return max_possible_score

        alpha = -100_000_000
        beta = min(beta, max_possible_score)

        for move in board.possible_moves():
            b = copy(board)
            b.play_move(move)
            score = -self.minimax(b, -beta, -alpha)
            alpha = max(alpha, score)
            if score >= beta:
                return alpha

        return alpha
