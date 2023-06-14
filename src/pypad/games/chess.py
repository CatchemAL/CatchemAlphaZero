from dataclasses import dataclass
from typing import Self

import chess
import numpy as np
from chess import Board, Move

from ..kaggle_types import Configuration, Observation
from ..states.state import Policy, State, Status
from ..views import View
from .chess_enums import ActionPlanes, ObsPlanes
from .game import Game, GameParameters

WHITE_IDXS = np.flipud(np.arange(64, dtype=np.uint64).reshape(8, 8))
WHITE_POWERS = 2**WHITE_IDXS

BLACK_IDXS = np.rot90(WHITE_IDXS, 2)
BLACK_POWERS = 2**BLACK_IDXS

PIECES = [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING]


@dataclass(slots=True)
class ChessState(State[Move]):
    board: Board

    @property
    def rows(self) -> int:
        return 8

    @property
    def cols(self) -> int:
        return 8

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    @property
    def played_by(self) -> int:
        return 1 + self.board.turn

    @property
    def move_count(self) -> int:
        return len(self.board.move_stack)

    def status(self) -> Status[int]:
        board = self.board
        if self.board.is_checkmate():
            return Status(False, self.played_by, 1.0, [])

        if board.is_fifty_moves() or board.is_insufficient_material() or board.is_repetition():
            return Status(False, self.played_by, 0.0, [])

        if board.is_repetition():
            return Status(False, self.played_by, 0.0, [])

        # Stalemate
        legal_moves = list(self.board.legal_moves)
        has_legal_moves = bool(legal_moves)
        return Status(has_legal_moves, self.played_by, 0.0, legal_moves)

    def play_move(self, move: Move) -> Self:
        new_board = self.board.copy()
        new_board.push(move)
        return ChessState(new_board)

    def set_move(self, move: Move) -> None:
        self.board.push(move)

    def policy_loc(self, move: Move) -> tuple[int, int, int]:
        _, rows, cols = ActionPlanes.shape()
        p, r, c = self.policy_loc_3d(move)
        idx = p * (rows * cols) + r * cols + c
        return idx

    def policy_loc_3d(self, move: Move) -> tuple[int, int, int]:
        if self.board.turn:
            from_idx, to_idx = move.from_square, move.to_square
        else:
            from_idx, to_idx = 63 - move.from_square, 63 - move.to_square

        x_from, x_to = from_idx % 8, to_idx % 8
        y_from, y_to = from_idx // 8, to_idx // 8
        x_shift = x_to - x_from
        y_shift = y_to - y_from

        if move.promotion and move.promotion != chess.QUEEN:
            match x_shift, y_shift, move.promotion:
                case (-1, 1, chess.KNIGHT):
                    return ActionPlanes.PROMOTE_KNIGHT_NW, 7 - y_from, x_from
                case (0, 1, chess.KNIGHT):
                    return ActionPlanes.PROMOTE_KNIGHT_N, 7 - y_from, x_from
                case (1, 1, chess.KNIGHT):
                    return ActionPlanes.PROMOTE_KNIGHT_NE, 7 - y_from, x_from
                case (-1, 1, chess.ROOK):
                    return ActionPlanes.PROMOTE_ROOK_NW, 7 - y_from, x_from
                case (0, 1, chess.ROOK):
                    return ActionPlanes.PROMOTE_ROOK_N, 7 - y_from, x_from
                case (1, 1, chess.ROOK):
                    return ActionPlanes.PROMOTE_ROOK_NE, 7 - y_from, x_from
                case (-1, 1, chess.BISHOP):
                    return ActionPlanes.PROMOTE_BISHOP_NW, 7 - y_from, x_from
                case (0, 1, chess.BISHOP):
                    return ActionPlanes.PROMOTE_BISHOP_N, 7 - y_from, x_from
                case (1, 1, chess.BISHOP):
                    return ActionPlanes.PROMOTE_BISHOP_NE, 7 - y_from, x_from
                case _:
                    raise ValueError("This should never happen")

        # N, NE, E, SE, S, SW, W, NW
        match x_shift, y_shift:
            case (0, y):
                if y_shift > 0:
                    return 0 + 8 * (y_shift - 1), 7 - y_from, x_from
                else:
                    return 4 + 8 * (-y_shift - 1), 7 - y_from, x_from
            case (x, 0):
                if x_shift > 0:
                    return 2 + 8 * (x_shift - 1), 7 - y_from, x_from
                else:
                    return 6 + 8 * (-x_shift - 1), 7 - y_from, x_from
            case (x, y) if x == y:
                if x_shift > 0:
                    return 1 + 8 * (x_shift - 1), 7 - y_from, x_from
                else:
                    return 5 + 8 * (-x_shift - 1), 7 - y_from, x_from
            case (x, y) if x == -y:
                if x_shift > 0:
                    return 3 + 8 * (x_shift - 1), 7 - y_from, x_from
                else:
                    return 7 + 8 * (-x_shift - 1), 7 - y_from, x_from
            case (1, 2):
                return ActionPlanes.KNIGHT_NNE, 7 - y_from, x_from
            case (2, 1):
                return ActionPlanes.KNIGHT_ENE, 7 - y_from, x_from
            case (2, -1):
                return ActionPlanes.KNIGHT_ESE, 7 - y_from, x_from
            case (1, -2):
                return ActionPlanes.KNIGHT_SSE, 7 - y_from, x_from
            case (-1, -2):
                return ActionPlanes.KNIGHT_SSW, 7 - y_from, x_from
            case (-2, -1):
                return ActionPlanes.KNIGHT_WSW, 7 - y_from, x_from
            case (-2, 1):
                return ActionPlanes.KNIGHT_WNW, 7 - y_from, x_from
            case (-1, 2):
                return ActionPlanes.KNIGHT_NNW, 7 - y_from, x_from

        raise ValueError("This should never happen")

    def to_feature(self) -> np.ndarray:
        PIECE_COUNT = 6
        COL = 1
        CASTLING = 2
        NO_PROG = 1
        EN_PASSANT = 1
        TWOFOLD = 1

        PLANES = 2 * PIECE_COUNT + COL + 2 * CASTLING + NO_PROG + EN_PASSANT + TWOFOLD

        can_white_queen_castle = np.sign(self.board.castling_rights & chess.BB_A1)
        can_white_king_castle = np.sign(self.board.castling_rights & chess.BB_H1)
        can_black_queen_castle = np.sign(self.board.castling_rights & chess.BB_A8)
        can_black_king_castle = np.sign(self.board.castling_rights & chess.BB_H8)

        if self.board.turn == chess.WHITE:
            player_idxs = WHITE_IDXS
            player_powers = WHITE_POWERS
            can_player_queenside = can_white_queen_castle
            can_player_kingside = can_white_king_castle
            can_opponent_queenside = can_black_queen_castle
            can_opponent_kingside = can_black_king_castle
        else:
            player_idxs = BLACK_IDXS
            player_powers = BLACK_POWERS
            can_player_queenside = can_black_queen_castle
            can_player_kingside = can_black_king_castle
            can_opponent_queenside = can_white_queen_castle
            can_opponent_kingside = can_white_king_castle

        feature = np.zeros((PLANES, 8, 8), dtype=np.float32)
        for i, piece in enumerate(PIECES):
            player_pieces = self.board.pieces_mask(piece, self.board.turn)
            feature[i, :, :] = np.sign(player_powers & player_pieces)

        for i, piece in enumerate(PIECES):
            opponent_pieces = self.board.pieces_mask(piece, not self.board.turn)
            feature[PIECE_COUNT + i, :, :] = np.sign(player_powers & opponent_pieces)

        is_twofold = self.board.is_repetition(2)

        feature[ObsPlanes.TURN, :, :] = self.board.turn
        feature[ObsPlanes.CAN_PLAYER_KINGSIDE, :, :] = can_player_kingside
        feature[ObsPlanes.CAN_PLAYER_QUEENSIDE, :, :] = can_player_queenside
        feature[ObsPlanes.CAN_OPP_KINGSIDE, :, :] = can_opponent_kingside
        feature[ObsPlanes.CAN_OPP_QUEENSIDE, :, :] = can_opponent_queenside
        feature[ObsPlanes.HALFMOVE_CLOCK, :, :] = self.board.halfmove_clock
        feature[ObsPlanes.EN_PASSANT_SQ, :, :] = self.board.ep_square == player_idxs
        feature[ObsPlanes.IS_TWOFOLD, :, :] = is_twofold

        return feature

    def to_grid(self) -> np.ndarray:
        ...

    def get_input_move(self) -> int:
        while True:
            response = input("Your turn to move. Please enter an integer: ")
            try:
                move = int(response)
                if move in self.status().legal_moves:
                    return move
            except ValueError:
                pass
            print("Move was invalid. Please try again.")

    def html(self, policy: np.ndarray | None = None, is_tiny_repr: bool = False) -> str:
        ...

    def plot(self, observation_plane: ObsPlanes) -> None:
        from ..views.plot import plot_chess_slice

        input_feature = self.to_feature()
        plot_chess_slice(input_feature, observation_plane)

    def plot_policy(self, action_plane: ActionPlanes, policy: Policy | None = None) -> None:
        from IPython.display import Math, display

        from ..views.plot import plot_chess_slice

        if policy is not None:
            value_expression = rf"v \in (-1, +1) = {policy.value:.3f}"
            display(Math(value_expression))

            encoded_policy = policy.encoded_policy
            vmax = encoded_policy.max()
            policy = encoded_policy.reshape(ActionPlanes.shape())
            plot_chess_slice(policy, action_plane, vmax=vmax)
        else:
            encoded_policy = np.zeros(ActionPlanes.shape())
            for move in self.status().legal_moves:
                p, r, c = self.policy_loc_3d(move)
                encoded_policy[p, r, c] = 1.0

            plot_chess_slice(encoded_policy, action_plane)

    def __copy__(self) -> "ChessState":
        board = self.board.copy()
        return ChessState(board)

    @classmethod
    def create(cls, fen: str | list[str] | None = None) -> "ChessState":
        board = Board()
        if fen:
            if isinstance(fen, str):
                board.set_fen(fen)
            elif isinstance(fen, list):
                for san in fen:
                    board.push_san(san)

        return ChessState(board)


class Chess(Game[ChessState]):
    ROWS, COLS = 8, 8

    def __init__(self, view: View[ChessState] | None = None) -> None:
        self.view = view

    @property
    def name(self) -> str:
        return "chess"

    @property
    def fullname(self) -> str:
        return "Chess"

    @property
    def shape(self) -> tuple[int, int]:
        return self.ROWS, self.COLS

    def initial_state(self, start: str | list[int] | None = None) -> ChessState:
        return ChessState.create(start)

    def config(self) -> GameParameters:
        shape = self.shape
        return GameParameters(shape, ObsPlanes.shape(), ActionPlanes.size())

    def from_kaggle(self, obs: Observation, config: Configuration) -> ChessState:
        pass

    def symmetries(
        self, encoded_state: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(encoded_state, policy)]

    def display(self, state: ChessState) -> None:
        pass

    def display_outcome(self, state: ChessState) -> None:
        pass
