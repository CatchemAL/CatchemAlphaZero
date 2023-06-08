from dataclasses import dataclass
from typing import Self

import chess
import numpy as np
from chess import Board, Move, Termination

from ..kaggle_types import Configuration, Observation
from ..states.state import State, Status, TemperatureSchedule
from ..views import View
from .game import Game, GameParameters


WHITE_IDXS = np.flipud(np.arange(64, dtype=np.uint64).reshape(8, 8))
WHITE_POWERS = 2**WHITE_IDXS

BLACK_IDXS = np.fliplr(np.arange(64, dtype=np.uint64).reshape(8, 8))
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

    def status(self) -> Status[int]:
        outcome = self.board.outcome()
        is_in_progress = outcome is None

        value = 1.0 if outcome.termination == Termination.CHECKMATE else 0.0
        legal_moves = list(self.board.legal_moves) if is_in_progress else []
        return Status(is_in_progress, self.played_by, value, legal_moves)

    def play_move(self, move: Move) -> Self:
        new_board = self.board.copy()
        new_board.push(move)
        return ChessState(new_board)

    def set_move(self, move: Move) -> None:
        self.board.push(move)

    def select_move(self, policy: np.ndarray, temperature_schedule: TemperatureSchedule) -> int:
        temperature = temperature_schedule.get_temperature(self.num_moves)
        if temperature < 0.001:
            return int(np.argmax(policy))

        temperature_policy = policy ** (1 / temperature)
        temperature_policy /= temperature_policy.sum()
        return int(np.random.choice(len(policy), p=temperature_policy))

    def to_feature(self) -> np.ndarray:
        PIECE_COUNT = 6
        COL = 1
        COUNT = 1
        CASTLING = 2
        NO_PROG = 1
        PLANES = 2 * PIECE_COUNT + COL + COUNT + 2 * CASTLING + NO_PROG

        is_white_queen_castle = np.sign(self.board.castling_rights & chess.BB_A1)
        is_white_king_castle = np.sign(self.board.castling_rights & chess.BB_H1)
        is_black_queen_castle = np.sign(self.board.castling_rights & chess.BB_A8)
        is_black_king_castle = np.sign(self.board.castling_rights & chess.BB_H8)

        if self.board.turn == chess.WHITE:
            player_powers = WHITE_POWERS
            is_player_queenside = is_white_queen_castle
            is_player_kingside = is_white_king_castle
            is_opponent_queenside = is_black_queen_castle
            is_opponent_kingside = is_black_king_castle
        else:
            player_powers = BLACK_POWERS
            is_player_queenside = is_black_queen_castle
            is_player_kingside = is_black_king_castle
            is_opponent_queenside = is_white_queen_castle
            is_opponent_kingside = is_white_king_castle

        feature = np.zeros((PLANES, 8, 8), dtype=np.float32)
        for i, piece in enumerate(PIECES):
            player_pieces = self.board.pieces_mask(piece, self.board.turn)
            feature[i, :, :] = np.sign(player_powers & player_pieces)

        for i, piece in enumerate(PIECES):
            opponent_pieces = self.board.pieces_mask(piece, not self.board.turn)
            feature[PIECE_COUNT + i, :, :] = np.sign(player_powers & opponent_pieces)

        feature[12, :, :] = self.board.turn
        feature[13, :, :] = len(self.board.move_stack)
        feature[14, :, :] = is_player_queenside
        feature[15, :, :] = is_player_kingside
        feature[16, :, :] = is_opponent_queenside
        feature[17, :, :] = is_opponent_kingside
        feature[18, :, :] = self.board.halfmove_clock

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

    def plot(self):
        ...

    def __copy__(self) -> "ChessState":
        board = self.board.copy()
        return ChessState(board)

    @classmethod
    def create(cls, fen: str | None = None) -> "ChessState":
        board = Board()
        if fen:
            board.set_fen(fen)

        return ChessState(board)


class Chess(Game[ChessState]):
    ROWS, COLS = 8, 8

    def __init__(self, view: View[ChessState] | None = None) -> None:
        self.view = view or ConsoleChessView()

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
        return Chess.create(start)

    def config(self) -> GameParameters:
        NUM_PLANES = 3

        shape = self.shape
        observation_shape = self.ROWS, self.COLS, NUM_PLANES
        action_size = self.ROWS * self.COLS
        return GameParameters(shape, observation_shape, action_size)

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
