import tkinter as tk

import chess
import numpy as np
from chess import Piece, PieceType, Square

from pypad.states import ChessState

ROWS, COLS = 8, 8

WHITE_IDXS = np.flipud(np.arange(64, dtype=np.uint64).reshape(8, 8))
WHITE_POWERS = 2**WHITE_IDXS

BLACK_IDXS = np.rot90(WHITE_IDXS, 2)
BLACK_POWERS = 2**BLACK_IDXS


class ChessGUI:
    CELL_SIZE = 80

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Play CatchemAlphaZero")
        self.state = ChessState.create()

        self.image_map = {
            Piece(chess.PAWN, chess.WHITE): tk.PhotoImage(file=r"icons/white-pawn.png"),
            Piece(chess.ROOK, chess.WHITE): tk.PhotoImage(file=r"icons/white-rook.png"),
            Piece(chess.KNIGHT, chess.WHITE): tk.PhotoImage(file=r"icons/white-knight.png"),
            Piece(chess.BISHOP, chess.WHITE): tk.PhotoImage(file=r"icons/white-bishop.png"),
            Piece(chess.QUEEN, chess.WHITE): tk.PhotoImage(file=r"icons/white-queen.png"),
            Piece(chess.KING, chess.WHITE): tk.PhotoImage(file=r"icons/white-king.png"),
            Piece(chess.PAWN, chess.BLACK): tk.PhotoImage(file=r"icons/black-pawn.png"),
            Piece(chess.ROOK, chess.BLACK): tk.PhotoImage(file=r"icons/black-rook.png"),
            Piece(chess.KNIGHT, chess.BLACK): tk.PhotoImage(file=r"icons/black-knight.png"),
            Piece(chess.BISHOP, chess.BLACK): tk.PhotoImage(file=r"icons/black-bishop.png"),
            Piece(chess.QUEEN, chess.BLACK): tk.PhotoImage(file=r"icons/black-queen.png"),
            Piece(chess.KING, chess.BLACK): tk.PhotoImage(file=r"icons/black-king.png"),
        }

        self.player_turn = "W"
        self.selected_square: Square | None = None

        # Create a canvas to draw the chess board and pieces
        self.canvas = tk.Canvas(self.window, width=700, height=700)
        self.canvas.pack()
        self.draw_board()

        # Bind mouse events to the canvas
        self.canvas.bind("<Button-1>", self.on_click)

        # Add labels to display game information
        self.turn_label = tk.Label(self.window, text=f"Turn: {self.player_turn}")
        self.turn_label.pack()
        self.status_label = tk.Label(self.window, text="Game in progress")
        self.status_label.pack()

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(ROWS):
            for j in range(COLS):
                x1 = j * self.CELL_SIZE
                y1 = i * self.CELL_SIZE
                x2 = x1 + self.CELL_SIZE
                y2 = y1 + self.CELL_SIZE
                color = "#EDF9EB" if (i + j) % 2 == 0 else "#346B51"
                rect = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline="", tags="my_rectangle"
                )

        self.draw_pieces()

    def draw_pieces(self):
        for rank in range(ROWS):
            for file in range(COLS):
                square = chess.square(file, 7 - rank)
                piece = self.state.board.piece_at(square)
                if piece:
                    self.canvas.create_image(
                        file * self.CELL_SIZE + self.CELL_SIZE // 2,
                        rank * self.CELL_SIZE + self.CELL_SIZE // 2,
                        image=self.image_map[piece],
                    )

    def on_click(self, event):
        file = event.x // self.CELL_SIZE
        rank = 7 - (event.y // self.CELL_SIZE)
        square = chess.square(file, rank)
        piece = self.state.board.piece_at(square)

        if self.selected_square is not None:
            if self.selected_square == square or self.move_piece(square):
                self.selected_square = None
                self.draw_board()
        elif piece and piece.color == self.state.board.turn:
            self.selected_square = square
            self.highlight_cell(file, rank)

    def highlight_cell(self, file, rank):
        x1 = file * self.CELL_SIZE
        y1 = (7 - rank) * self.CELL_SIZE
        x2 = x1 + self.CELL_SIZE
        y2 = y1 + self.CELL_SIZE

        highlighted_cell = self.canvas.create_rectangle(
            x1 + 2,
            y1 + 2,
            x2 - 2,
            y2 - 2,
            fill="#CEF261",
            outline="#CEF261",
            width=4,
            tag="highlight",
        )
        self.canvas.tag_lower(highlighted_cell, "all")
        self.canvas.tag_raise(highlighted_cell, "my_rectangle")

    def remove_highlight(self):
        self.canvas.delete("highlight")

    def move_piece(self, to_square: Square):
        from_square = self.selected_square
        piece = self.state.board.piece_at(from_square)
        move = chess.Move(from_square, to_square)

        if piece.piece_type == chess.PAWN and chess.square_rank(to_square) == 7:
            move.promotion = chess.QUEEN

        # Check if the move is legal
        if piece.color == self.state.board.turn:
            if move in self.state.board.legal_moves:
                self.state.board.push(move)
                return True

        return False


if __name__ == "__main__":
    # Create a new ChessGUI object and start the game
    game = ChessGUI()
    game.window.mainloop()
