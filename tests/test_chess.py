import numpy as np
import pytest
from caz.games import Chess

from caz.states.chess import ObsPlanes
from caz.states.chess_enums import KeyGames


class TestChessFeatures:
    @pytest.mark.parametrize(
        "label,turn,can_player_kingside,can_player_queenside,can_opp_kingside,can_opp_queenside,half_move_clock,ep_square",
        [
            ("white_en_passant_sans", 1, 1, 1, 1, 1, 0, 1),
            ("black_en_passant_sans", 0, 1, 1, 1, 1, 0, 1),
            ("white_no_queenside_sans", 1, 1, 0, 1, 1, 0, 1),
            ("white_no_kingside_sans", 0, 1, 1, 0, 1, 1, 0),
            ("white_no_castling_sans", 1, 0, 0, 1, 1, 0, 1),
            ("black_no_queenside_sans", 0, 1, 0, 1, 1, 4, 0),
            ("black_no_kingside_sans", 0, 0, 1, 1, 1, 4, 0),
            ("black_no_castling_sans", 1, 1, 1, 0, 0, 4, 0),
            ("white_castled_black_not_yet_sans", 1, 0, 0, 1, 1, 0, 0),
        ],
    )
    def test_feature_creation(
        self,
        label: str,
        turn: float,
        can_player_kingside: float,
        can_player_queenside: float,
        can_opp_kingside: float,
        can_opp_queenside: float,
        half_move_clock: float,
        ep_square: float | None,
    ) -> None:
        # Arrange
        game = Chess()
        sans = KeyGames.get(label)
        sut = game.initial_state(sans)

        # Act
        actual = sut.to_feature()

        # Assert
        assert np.all(actual[ObsPlanes.TURN, :, :] == turn)
        assert np.all(actual[ObsPlanes.CAN_PLAYER_KINGSIDE, :, :] == can_player_kingside)
        assert np.all(actual[ObsPlanes.CAN_PLAYER_QUEENSIDE, :, :] == can_player_queenside)
        assert np.all(actual[ObsPlanes.CAN_OPP_KINGSIDE, :, :] == can_opp_kingside)
        assert np.all(actual[ObsPlanes.CAN_OPP_QUEENSIDE, :, :] == can_opp_queenside)
        assert np.all(actual[ObsPlanes.HALFMOVE_CLOCK, :, :] == half_move_clock)
        assert np.sum(actual[ObsPlanes.EN_PASSANT_SQ, :, :]) == ep_square
