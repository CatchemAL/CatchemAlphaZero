from enum import Enum

from .game import Chess, ConnectX, TicTacToe


class GameType(Enum):
    """Enum representing the types of games supported."""

    TICTACTOE = "TICTACTOE"
    CONNECTX = "CONNECTX"
    CHESS = "CHESS"

    @staticmethod
    def from_str(value: str) -> "GameType":
        """Converts a string to its enum representation.

        Args:
            value (str): The string representation of the game type.

        Raises:
            ValueError: If the string is not recognised.

        Returns:
            GameType: The enum
        """
        if value.upper().startswith("TIC"):
            return GameType.TICTACTOE
        if value.upper() == "CONNECTX":
            return GameType.CONNECTX
        if value.upper() == "CHESS":
            return GameType.CHESS
        supported_types = ", ".join([e.name for e in GameType])
        message = f"{value} not a supported solver type. Supported types are {supported_types}."
        raise ValueError(message)

    def create(self) -> TicTacToe | ConnectX:
        match self:
            case GameType.TICTACTOE:
                return TicTacToe()
            case GameType.CONNECTX:
                return ConnectX()
            case _:
                raise ValueError(f"Invalid game type: {self}")
