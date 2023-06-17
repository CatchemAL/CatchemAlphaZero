import tkinter as tk
from tkinter import ttk

from PIL import ImageTk, Image

import chess
from chess import Piece, Square, PieceType
from pypad.states import ChessState

ROWS, COLS = 8, 8
CELL_SIZE = 80

DARK_COLOUR = "#1F1F1F"
TEAL_COLOUR = "#78C5B0"
BLUE_COLOUR = "#77BEFC"
HIGH_COLOUR = "#B5E61D"


class ChessScreen(tk.Frame):
    def __init__(self, master, switch_screen_callback):
        super().__init__(master, bg=DARK_COLOUR)

        self.state = ChessState.create()

        self.selected_square: Square | None = None
        self.image_map = ChessScreen._get_piece_image_map()

        # LHS Canvas
        self.canvas = tk.Canvas(
            self,
            width=CELL_SIZE * COLS,
            height=CELL_SIZE * ROWS,
            bg="white",
            border=0,
        )
        self.canvas.pack(side=tk.LEFT, padx=25, pady=25, expand=False)
        self.canvas.bind("<Button-1>", self.on_click)
        self.draw_board()

        # RHS Frame
        rhs_frame = tk.Frame(self, bg=DARK_COLOUR)
        rhs_frame.pack(side=tk.LEFT, padx=(0, 20), anchor=tk.W, expand=True)

        # Add labels to display game information
        self.status_label = tk.Label(
            rhs_frame,
            text="Status",
            font=("Cascadia Mono", 12),
            fg=TEAL_COLOUR,
            bg=DARK_COLOUR,
        )
        self.status_label.pack(side=tk.TOP, pady=0, anchor=tk.W)

        # Add labels to display game information
        self.white_player_label = tk.Label(
            rhs_frame,
            text="White: Human",
            font=("Cascadia Mono", 11),
            fg=BLUE_COLOUR,
            bg=DARK_COLOUR,
        )
        self.white_player_label.pack(side=tk.TOP, pady=(0, 0), anchor=tk.W)

        # Add labels to display game information
        self.black_player_label = tk.Label(
            rhs_frame,
            text="Black: CatchemAlpha",
            font=("Cascadia Mono", 11),
            fg=BLUE_COLOUR,
            bg=DARK_COLOUR,
        )
        self.black_player_label.pack(side=tk.TOP, pady=0, anchor=tk.W)

        # Add labels to display game information
        self.outcome_label = tk.Label(
            rhs_frame,
            text="Game:  In Progress",
            font=("Cascadia Mono", 11),
            fg=BLUE_COLOUR,
            bg=DARK_COLOUR,
        )
        self.outcome_label.pack(side=tk.TOP, pady=0, anchor=tk.W)

        # Create radio buttons for promotion options
        self.promotion_option = tk.IntVar(value=chess.QUEEN)
        self._add_promotion_options(rhs_frame)

        # Load and display the image
        image_path = "icons/ui_image.png"  # Replace with the actual path to your PNG image
        image = Image.open(image_path)
        image = image.resize((180, 176))  # Adjust the size of the image as needed
        photo = ImageTk.PhotoImage(image)
        image_label = tk.Label(rhs_frame, image=photo, bg=DARK_COLOUR)
        image_label.image = photo  # Store a reference to avoid garbage collection
        image_label.pack(side=tk.TOP, pady=(60, 30), anchor=tk.W)

        # Create a custom style for the buttons
        style = ttk.Style()
        style.theme_use("clam")  # put the theme name here, that you want to use
        style.configure(
            "W.TButton",
            foreground=DARK_COLOUR,
            background=BLUE_COLOUR,
            font=("Cascadia Mono", 12),
            relief=tk.FLAT,
            anchor=tk.CENTER,
        )

        # Create the "Go to Title Screen" button
        new_game_button = ttk.Button(
            rhs_frame,
            text="New Game",
            command=switch_screen_callback,
            style="W.TButton",
            compound=tk.CENTER,
        )
        new_game_button.pack(side=tk.BOTTOM, anchor=tk.S, pady=(10, 10), padx=(0, 0))

    def draw_board(self):
        self.canvas.delete("all")
        for rank in range(ROWS):
            for file in range(COLS):
                x1, y1 = file * CELL_SIZE, rank * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                color = "#EDF9EB" if (rank + file) % 2 == 0 else "#346B51"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="", tags="cell_rect")

        for rank in range(ROWS):
            for file in range(COLS):
                square = chess.square(file, 7 - rank)
                piece = self.state.board.piece_at(square)
                if piece:
                    self.canvas.create_image(
                        file * CELL_SIZE + CELL_SIZE // 2,
                        rank * CELL_SIZE + CELL_SIZE // 2,
                        image=self.image_map[piece],
                    )

    def on_click(self, event):
        file = event.x // CELL_SIZE
        rank = 7 - (event.y // CELL_SIZE)
        square = chess.square(file, rank)
        piece = self.state.board.piece_at(square)

        if piece and piece.color == self.state.board.turn and square != self.selected_square:
            self.selected_square = square
            self.highlight_cell(file, rank)

        elif self.selected_square is not None:
            self.move_piece(square)
            self.selected_square = None
            self.draw_board()

    def highlight_cell(self, file, rank):
        x1 = file * CELL_SIZE
        y1 = (7 - rank) * CELL_SIZE
        x2 = x1 + CELL_SIZE
        y2 = y1 + CELL_SIZE

        self.canvas.delete("highlight")
        highlighted_cell = self.canvas.create_rectangle(
            x1 + 2,
            y1 + 2,
            x2 - 2,
            y2 - 2,
            fill=HIGH_COLOUR,
            outline=HIGH_COLOUR,
            width=4,
            tag="highlight",
        )
        # self.canvas.tag_lower(highlighted_cell, "all")
        self.canvas.tag_raise(highlighted_cell, "cell_rect")

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

    def _add_promotion_options(self, frame) -> None:
        piece_map = {
            "Queen": chess.QUEEN,
            "Knight": chess.KNIGHT,
            "Rook": chess.ROOK,
            "Bishop": chess.BISHOP,
        }

        self.status_label = tk.Label(
            frame,
            text="Promotion Piece",
            font=("Cascadia Mono", 12),
            fg=TEAL_COLOUR,
            bg=DARK_COLOUR,
        )
        self.status_label.pack(side=tk.TOP, anchor=tk.W, pady=(60, 0))

        for text, value in piece_map.items():
            radio = tk.Radiobutton(
                frame,
                text=text,
                variable=self.promotion_option,
                value=value,
                font=("Cascadia Mono", 11),
                bg=DARK_COLOUR,
                fg=BLUE_COLOUR,
            )
            radio.pack(anchor=tk.W, side=tk.TOP)

    @staticmethod
    def _get_piece_image_map() -> dict[Piece, tk.PhotoImage]:
        return {
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


class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("CatchemAlphaZero")
        self.resizable(False, False)
        self.iconbitmap("icons/tiny_chess.ico")
        self.game_screen = ChessScreen(self, lambda x: x)
        self.game_screen.pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    app = Application()
    app.mainloop()
