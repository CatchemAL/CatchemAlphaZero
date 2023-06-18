from __future__ import annotations

import asyncio
import tkinter as tk
from asyncio import Event
from tkinter import ttk

import chess
from chess import Piece, Square
from PIL import Image, ImageTk

from pypad.games import Chess
from pypad.solvers.alpha_zero import AlphaZero
from pypad.solvers.network_torch import PytorchNeuralNetwork
from pypad.states import ChessState
from pypad.states.state import TemperatureSchedule

ROWS, COLS = 8, 8
CELL_SIZE = 80
REFRESH_RATE = 0.025

BLACK_COLOR = "#EDF9EB"
WHITE_COLOR = "#346B51"

DARK_COLOR = "#1F1F1F"
TEAL_COLOR = "#78C5B0"
BLUE_COLOR = "#77BEFC"
MATE_COLOR = "#880015"
DRAW_COLOR = "#FFA628"
LAST_COLOR_L = "#F8F77E"
LAST_COLOR_D = "#BECC52"
HIGH_COLOR = "#B5E61D"


class ChessScreenModel:
    def __init__(self, alpha_zero: AlphaZero, colors: list[chess.Color]) -> None:
        self.alpha_zero = alpha_zero
        self.colors = colors
        self.root_node = None
        self.event: Event | None = None

    def is_alphas_turn(self, state: ChessState) -> bool:
        return state.board.turn in self.colors

    def reset(self, colors: list[chess.Color]) -> None:
        self.root_node = None
        self.colors = colors

    async def ponder(self, state: ChessState) -> None:
        if not self.colors:
            return

        PONDER_SIMS = 50_000
        await asyncio.sleep(REFRESH_RATE)
        self.event = Event()
        policy = await self.alpha_zero.policy_async(state, PONDER_SIMS, self.root_node, self.event)
        self.root_node = policy.node

    async def stop_pondering(self, move: chess.Move) -> None:
        if self.event:
            self.event.set()
            self.event = None

    async def get_move(self, state: ChessState, num_sims: int) -> chess.Move:
        node = None
        if len(self.colors) == 2:
            node = self.root_node
        elif self.root_node and state.board.move_stack:
            node = next(n for n in self.root_node.children if n.move == state.board.move_stack[-1])

        temperature_schedule = TemperatureSchedule.competitive()
        await asyncio.sleep(REFRESH_RATE)
        policy = await self.alpha_zero.policy_async(state, num_sims, node)
        temperature = temperature_schedule.get_temperature(state.move_count)
        move = policy.select_move(temperature)
        self.root_node = next(node for node in policy.node.children if node.move == move)
        return move


class ChessScreenController:
    def __init__(self, model: ChessScreenModel, view: ChessScreen):
        self.model = model
        self.view = view

        self.state = ChessState.create()

        self.selected_square: Square | None = None

        self.view.on_square_click = self.on_square_click
        self.view.draw_board(self.state)

    @property
    def num_sims(self) -> int:
        return int(self.view.sims_combobox.get())

    async def reset(self, human_colors: list[chess.Color]) -> None:
        colors_complement = []
        if chess.WHITE not in human_colors:
            colors_complement.append(chess.WHITE)
        if chess.BLACK not in human_colors:
            colors_complement.append(chess.BLACK)

        self.state = ChessState.create()
        self.model.reset(colors_complement)
        self.view.set_players("Human", human_colors)
        self.view.set_players("CAZ", colors_complement)
        self.view.draw_board(self.state)
        if self.state.status().is_in_progress and self.model.is_alphas_turn(self.state):
            move = await self.model.get_move(self.state, self.num_sims)
            await self.move_piece(move)

    async def on_square_click(self, file: int, rank: int) -> None:
        if self.model.is_alphas_turn(self.state) or not self.state.status().is_in_progress:
            return

        square = chess.square(file, rank)
        piece = self.state.board.piece_at(square)

        if piece and piece.color == self.state.board.turn and square != self.selected_square:
            self.selected_square = square
            self.view.highlight_cell(file, rank, HIGH_COLOR, True)

        elif self.selected_square is not None:
            move = chess.Move(self.selected_square, square)
            await self.move_piece(move)

        else:
            self.view.draw_board(self.state)

    async def move_piece(self, move: chess.Move):
        from_square = move.from_square
        to_square = move.to_square
        piece = self.state.board.piece_at(from_square)
        self.selected_square = None

        if move.promotion is None and piece.piece_type == chess.PAWN:
            promotion_rank = 7 if self.state.board.turn else 0
            if chess.square_rank(to_square) == promotion_rank:
                move.promotion = self.view.promotion_option.get()

        # Check if the move is legal
        if piece.color == self.state.board.turn and move in self.state.board.legal_moves:
            await self.model.stop_pondering(move)
            self.state.board.push(move)
            self.view.draw_board(self.state)

            if not self.state.status().is_in_progress:
                return

            if self.model.is_alphas_turn(self.state):
                move = await self.model.get_move(self.state, self.num_sims)
                await self.move_piece(move)
            else:
                await self.model.ponder(self.state)
        else:
            self.view.draw_board(self.state)


class ChessScreen(tk.Frame):
    def __init__(self, parent, new_game_callback):
        super().__init__(parent, bg=DARK_COLOR)

        # Initialize variables
        self.image_map = ChessScreen._get_piece_image_map()

        # LHS Canvas
        self.canvas = tk.Canvas(self, width=8 * CELL_SIZE, height=8 * CELL_SIZE, bg="white", border=0)
        self.canvas.pack(side=tk.LEFT, padx=25, pady=25, expand=False)
        self.canvas.bind("<Button-1>", self._canvas_callback)

        # RHS Frame
        rhs_frame = tk.Frame(self, bg=DARK_COLOR)
        rhs_frame.pack(side=tk.LEFT, padx=(0, 20), anchor=tk.W, expand=True)

        # Add labels to display game information
        self.white_label, self.black_label = self._add_status_labels(rhs_frame)

        # Create radio buttons for promotion options
        self.promotion_option = tk.IntVar(value=chess.QUEEN)
        self._add_promotion_options(rhs_frame)

        # Add new game button
        self._add_catchemalphazero_logo(rhs_frame)
        self._add_new_game_button(rhs_frame, new_game_callback)

    async def on_square_click(self, file: int, rank: int) -> None:
        ...

    def draw_board(self, state: ChessState):
        self.canvas.delete("all")

        # Big white square
        self.canvas.create_rectangle(
            0,
            0,
            8 * CELL_SIZE,
            8 * CELL_SIZE,
            fill=WHITE_COLOR,
            outline="",
            tags="cell_rect",
        )

        # Overlay the black squares
        for square in range(ROWS * COLS):
            file, rank = chess.square_file(square), chess.square_rank(square)
            if (file + rank) % 2 == 1:
                continue
            x1, y1 = file * CELL_SIZE, rank * CELL_SIZE
            x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=BLACK_COLOR, outline="", tags="cell_rect")

        for rank in range(ROWS):
            for file in range(COLS):
                square = chess.square(file, 7 - rank)
                piece = state.board.piece_at(square)
                if piece:
                    self.canvas.create_image(
                        file * CELL_SIZE + CELL_SIZE // 2,
                        rank * CELL_SIZE + CELL_SIZE // 2,
                        image=self.image_map[piece],
                    )

        if state.board.move_stack:
            last_move = state.board.move_stack[-1]
            self.highlight_last_move(last_move)

        status = state.status()
        if not status.is_in_progress:
            self.highlight_mate(state) if status.value > 0 else self.highlight_draw(state)

    def highlight_last_move(self, move: chess.Move) -> None:
        from_square = move.from_square
        file, rank = chess.square_file(from_square), chess.square_rank(from_square)
        color = LAST_COLOR_L if (file + rank) & 1 else LAST_COLOR_D
        self.highlight_cell(file, rank, color, True, "move_highlight")

        to_square = move.to_square
        file, rank = chess.square_file(to_square), chess.square_rank(to_square)
        color = LAST_COLOR_L if (file + rank) & 1 else LAST_COLOR_D
        self.highlight_cell(file, rank, color, False, "move_highlight")

    def highlight_draw(self, state: ChessState) -> None:
        king_square = state.board.king(state.board.turn)
        file, rank = chess.square_file(king_square), chess.square_rank(king_square)
        self.highlight_cell(file, rank, DRAW_COLOR, True)

        king_square = state.board.king(not state.board.turn)
        file, rank = chess.square_file(king_square), chess.square_rank(king_square)
        self.highlight_cell(file, rank, DRAW_COLOR, False)

    def highlight_mate(self, state: ChessState) -> None:
        king_square = state.board.king(state.board.turn)
        file, rank = chess.square_file(king_square), chess.square_rank(king_square)
        self.highlight_cell(file, rank, MATE_COLOR, True)

    def highlight_cell(
        self, file: int, rank: int, color: str, delete_canvas: bool, tag: str = "highlight"
    ):
        x1 = file * CELL_SIZE
        y1 = (7 - rank) * CELL_SIZE
        x2 = x1 + CELL_SIZE
        y2 = y1 + CELL_SIZE

        if delete_canvas:
            self.canvas.delete(tag)
        highlighted_cell = self.canvas.create_rectangle(
            x1 + 2,
            y1 + 2,
            x2 - 2,
            y2 - 2,
            fill=color,
            outline=color,
            width=4,
            tag=tag,
        )
        self.canvas.tag_raise(highlighted_cell, "cell_rect")

    def set_players(self, label: str, colors: list[chess.Color]) -> None:
        if chess.WHITE in colors:
            self.white_label.config(text=f"White: {label}")
        if chess.BLACK in colors:
            self.black_label.config(text=f"Black: {label}")

    def _add_status_labels(self, frame) -> tk.Label:
        players_label = tk.Label(
            frame,
            text="Status",
            font=("Cascadia Mono", 12),
            fg=TEAL_COLOR,
            bg=DARK_COLOR,
        )
        players_label.pack(side=tk.TOP, pady=0, anchor=tk.W)

        white_player_label = tk.Label(
            frame,
            text=f"White: <<PLACEHOLDER>>",
            font=("Cascadia Mono", 11),
            fg=BLUE_COLOR,
            bg=DARK_COLOR,
        )
        white_player_label.pack(side=tk.TOP, pady=(0, 0), anchor=tk.W)

        black_player_label = tk.Label(
            frame,
            text=f"Black: <<PLACEHOLDER>>",
            font=("Cascadia Mono", 11),
            fg=BLUE_COLOR,
            bg=DARK_COLOR,
        )
        black_player_label.pack(side=tk.TOP, pady=0, anchor=tk.W)

        return white_player_label, black_player_label

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
            fg=TEAL_COLOR,
            bg=DARK_COLOR,
        )
        self.status_label.pack(side=tk.TOP, anchor=tk.W, pady=(20, 0))

        for text, value in piece_map.items():
            radio = tk.Radiobutton(
                frame,
                text=text,
                variable=self.promotion_option,
                value=value,
                font=("Cascadia Mono", 11),
                bg=DARK_COLOR,
                fg=BLUE_COLOR,
            )
            radio.pack(anchor=tk.W, side=tk.TOP)

        self.simluation_label = tk.Label(
            frame,
            text="MCTS Num Sims",
            font=("Cascadia Mono", 12),
            fg=TEAL_COLOR,
            bg=DARK_COLOR,
        )
        self.simluation_label.pack(side=tk.TOP, anchor=tk.W, pady=(20, 0))

        # Create a Combobox
        values = (10, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
        self.sims_combobox = ttk.Combobox(frame, values=values, state="readonly")
        self.sims_combobox.current(4)
        self.sims_combobox.pack(side=tk.TOP, anchor=tk.W, pady=(10, 0))

    def _add_catchemalphazero_logo(self, frame) -> None:
        # Load and display the image
        image_path = "icons/caz_flat_logo.png"
        image = Image.open(image_path)
        image = image.resize((180, 176))
        photo = ImageTk.PhotoImage(image)
        image_label = tk.Label(frame, image=photo, bg=DARK_COLOR)
        image_label.image = photo  # Store a reference to avoid garbage collection
        image_label.pack(side=tk.TOP, pady=(60, 30), anchor=tk.W)

    def _add_new_game_button(self, frame, new_game_callback) -> None:
        # Create a custom style for the buttons
        style = ttk.Style()
        style.theme_use("clam")  # put the theme name here, that you want to use
        style.configure(
            "W.TButton",
            foreground=DARK_COLOR,
            background=BLUE_COLOR,
            font=("Cascadia Mono", 12),
            relief=tk.FLAT,
            anchor=tk.CENTER,
        )

        # Create the "Go to Title Screen" button
        new_game_button = ttk.Button(
            frame,
            text="New Game",
            command=new_game_callback,
            style="W.TButton",
            compound=tk.CENTER,
        )
        new_game_button.pack(side=tk.BOTTOM, anchor=tk.S, pady=(10, 10), padx=(0, 0))

    def _canvas_callback(self, event) -> None:
        file = event.x // CELL_SIZE
        rank = 7 - (event.y // CELL_SIZE)
        asyncio.create_task(self.on_square_click(file, rank))

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
        self.is_open = True

        game = Chess()
        network = PytorchNeuralNetwork.create(game, ".")
        alpha_zero = AlphaZero(network)
        model = ChessScreenModel(alpha_zero, [chess.BLACK])

        self.game_screen = ChessScreen(self, lambda: print("Single Window!"))
        self.controller = ChessScreenController(model, self.game_screen)
        self.game_screen.pack(fill=tk.BOTH, expand=True)

        self.protocol("WM_DELETE_WINDOW", self.raise_exit_flag)

    async def show(self):
        while self.is_open:
            self.update()
            await asyncio.sleep(0.025)

    def raise_exit_flag(self):
        self.is_open = False


if __name__ == "__main__":
    app = Application()
    asyncio.run(app.show())
