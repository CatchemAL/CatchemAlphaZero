from copy import copy

from .state import ConnectX


class Solver:
    def minimax(self, state: ConnectX, alpha: int, beta: int) -> int:
        if state.is_full():
            return 0

        win_mask = state.win_mask()
        possible_moves = state.possible_moves_mask()
        if win_mask & possible_moves:
            return (state.num_slots - state.num_moves + 1) // 2

        max_possible_score = (state.num_slots - state.num_moves - 1) // 2
        if max_possible_score <= alpha:
            return max_possible_score

        alpha = -100_000_000
        beta = min(beta, max_possible_score)

        for move in state.possible_moves():
            b = copy(state)
            b.play_move(move)
            score = -self.minimax(b, -beta, -alpha)
            alpha = max(alpha, score)
            if score >= beta:
                return alpha

        return alpha
