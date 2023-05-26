from copy import copy

from ..states import ConnectXState


class Solver:
    def minimax(self, state: ConnectXState, alpha: int, beta: int) -> int:
        if state.is_full():
            return 0

        win_mask = state.win_mask()
        possible_moves = state._possible_bitmoves_mask()
        if win_mask & possible_moves:
            return (state.num_slots - state.num_moves + 1) // 2

        max_possible_score = (state.num_slots - state.num_moves - 1) // 2
        if max_possible_score <= alpha:
            return max_possible_score

        alpha = -100_000_000
        beta = min(beta, max_possible_score)

        for move in state.possible_bitmoves():
            b = copy(state)
            b.play_bitmove(move)
            score = -self.minimax(b, -beta, -alpha)
            alpha = max(alpha, score)
            if score >= beta:
                return alpha

        return alpha
