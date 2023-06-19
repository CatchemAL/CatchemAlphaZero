from colorama import Fore, Style

from ..states import ChessState, ConnectXState, TicTacToeState
from . import View


class ConsoleConnectXView(View[ConnectXState]):
    def display(self, state: ConnectXState) -> None:
        grid = state.to_grid()
        num_rows, num_cols = grid.shape

        color_map = {0: Fore.WHITE, 1: Fore.RED, 2: Fore.YELLOW}

        # Build the string representation
        output = ""
        for row in range(num_rows):
            for col in range(num_cols):
                cell = grid[row, col]
                color = color_map.get(cell, Fore.RESET)
                output += f"{color}{cell} "
            output += "\n"

        output += Style.RESET_ALL

        print(output)

    def display_outcome(self, state: ConnectXState) -> None:
        if state.is_won():
            print(f"Player {state.played_by}  wins!")
        else:
            print("It's a draw!")


class ConsoleTicTacToeView(View[TicTacToeState]):
    def display(self, state: TicTacToeState) -> None:
        grid = state.to_grid()
        print(grid)

    def display_outcome(self, state: TicTacToeState) -> None:
        if state.is_won():
            print(f"Player {state.played_by}  wins!")
        else:
            print("It's a draw!")


class ChessTicTacToeView(View[ChessState]):
    def display(self, state: ChessState) -> None:
        print(state.board)
        print()

    def display_outcome(self, state: ChessState) -> None:
        if state.status().value:
            print(f"Player {state.played_by}  wins!")
        else:
            print("It's a draw!")
