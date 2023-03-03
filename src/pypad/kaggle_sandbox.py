import numpy as np
import random
from functools import partial

from kaggle_environments import make

from .connectx import Board
from .solver import Solver



# Helper function for score_move: calculates value of heuristic for grid
def get_heuristic(grid, mark, config):
    num_threes = count_windows(grid, 3, mark, config)
    num_fours = count_windows(grid, 4, mark, config)
    num_threes_opp = count_windows(grid, 3, mark%2+1, config)
    score = num_threes - 1e2*num_threes_opp + 1e6*num_fours
    return score

def get_heuristic2(board) -> int:
    directions = (board.mask_utils.rows - 1, board.mask_utils.rows, board.mask_utils.rows + 1, 1)
    bitboard = board.position ^ board.mask;
    count = 0
    for dir in directions:
        if bitboard & (bitboard >> dir) & (bitboard >> 2 * dir):
            count += 0.1

    return count


def shallow_negamax(board: Board, alpha: int, beta: int, depth: int) -> int:
    if board.is_full():
        return 0

    win_mask = board.win_mask()
    possible_moves = board.possible_moves_mask()
    if (win_mask & possible_moves):
        return (board.num_slots() - board.num_moves + 1) // 2

    max_possible_score = (board.num_slots() - board.num_moves - 1) // 2
    if max_possible_score <= alpha:
        return max_possible_score
    
    if depth == 0:
        return get_heuristic2(board)

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

def agent_negamax(obs, config):

    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    board = Board.from_grid(grid)

    best_col, best_score = next(board.possible_col_moves()), -1_000_000
    for col in board.possible_col_moves():
        b = board.copy()
        b.play_col(col)
        alpha, beta, depth = -1, 1, 6
        score = -shallow_negamax(b, alpha, beta, depth)
        if score > best_score:
            best_score = score
            best_col = col
            
    return best_col

def agent_negamax2(obs, config):

    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    board = Board.from_grid(grid)

    best_col, best_score = next(board.possible_col_moves()), -1_000_000
    for col in board.possible_col_moves():
        b = board.copy()
        b.play_col(col)
        alpha, beta, depth = -1, 1, 4
        score = -shallow_negamax(b, alpha, beta, depth)
        if score >= best_score:
            best_score = score
            best_col = col
            
    return best_col

def run_kaggle() -> None:
    # Setup a tictactoe environment.
    env = make("connectx", debug=True)
    env.run([agent_negamax2, agent_negamax])
    env.render(mode="ipython")