import numpy as np
from functools import partial

from kaggle_environments import make

from .connectx import Board


# Helper function for score_move: calculates value of heuristic for grid
def get_heuristic(board) -> int:
    directions = (board.mask_utils.rows - 1, board.mask_utils.rows, board.mask_utils.rows + 1, 1)
    bitboard = board.position ^ board.mask
    bitboard2 = board.position
    score = 0
    for dir in directions:
        if result := bitboard & (bitboard >> dir) & (bitboard >> 2 * dir):
            score += 0.1 * result.bit_count()

        if result := bitboard2 & (bitboard2 >> dir) & (bitboard2 >> 2 * dir):
            score -= 0.2 * result.bit_count()

    return score


def shallow_negamax(board: Board, alpha: int, beta: int, depth: int) -> int:
    if board.is_full():
        return 0

    win_mask = board.win_mask()
    possible_moves = board.possible_moves_mask()
    if win_mask & possible_moves:
        return (board.num_slots() - board.num_moves + 1) // 2

    max_possible_score = (board.num_slots() - board.num_moves - 1) // 2
    if max_possible_score <= alpha:
        return max_possible_score

    if depth == 0:
        return get_heuristic(board)

    alpha = -100_000_000
    beta = min(beta, max_possible_score)

    for move in board.possible_moves():
        b = board.copy()
        b.play_move(move)
        score = -shallow_negamax(b, -beta, -alpha, depth - 1)
        alpha = max(alpha, score)
        if score >= beta:
            return alpha

    return alpha


def agent_negamax(obs, config, depth):
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    board = Board.from_grid(grid)

    best_col, best_score = next(board.possible_col_moves()), -1_000_000

    for col in board.possible_col_moves():
        b = board.copy()
        b.play_col(col)
        if b.is_won():
            return col

    for col in board.possible_col_moves():
        b = board.copy()
        b.play_col(col)
        alpha, beta = -1, 1
        score = -shallow_negamax(b, alpha, beta, depth)
        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def run_kaggle() -> None:
    agent_negamax5 = partial(agent_negamax, depth=5)
    agent_negamax2 = partial(agent_negamax, depth=2)

    # Setup a tictactoe environment.
    env = make("connectx", debug=True)
    env.run([agent_negamax2, agent_negamax5])
    env.render(mode="ipython")
