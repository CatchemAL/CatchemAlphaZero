from __future__ import annotations

import asyncio
import tkinter as tk
from tkinter import filedialog, ttk

import chess
from PIL import Image, ImageTk

from pypad.games import Chess
from pypad.solvers.alpha_zero import AlphaZero
from pypad.solvers.network_torch import PytorchNeuralNetwork
from pypad.views.chess_detail import (
    DARK_COLOR,
    REFRESH_RATE,
    ChessScreen,
    ChessScreenController,
    ChessScreenModel,
)

GREY_COLOR = "#BEC7C6"


class Application(tk.Tk):
    def __init__(self, event_loop):
        super().__init__()

        self.loop = event_loop
        self.is_open = True
        self.title("CatchemAlphaZero")
        self.resizable(False, False)
        self.iconbitmap("icons/tiny_chess.ico")
        self.current_screen = None
        self.protocol("WM_DELETE_WINDOW", self.raise_exit_flag)

        game = Chess()
        network = PytorchNeuralNetwork.create(game, ".")
        alpha_zero = AlphaZero(network)
        model = ChessScreenModel(alpha_zero, chess.BLACK)

        self.title_screen = TitleScreen(self, self.switch_to_game_screen)
        self.game_screen = ChessScreen(self, self.switch_to_title_screen)
        self.game_controller = ChessScreenController(model, self.game_screen)

        self.switch_to_title_screen()
        self.create_menu()

    async def show_async(self):
        while self.is_open:
            self.update()
            await asyncio.sleep(REFRESH_RATE)

    def raise_exit_flag(self):
        self.is_open = False

    def switch_to_title_screen(self):
        if self.current_screen:
            self.current_screen.pack_forget()
        self.title_screen.pack(fill=tk.BOTH, expand=True)
        self.current_screen = self.title_screen

    def switch_to_game_screen(self, human_colors: list[chess.Color]) -> None:
        if self.current_screen:
            self.current_screen.pack_forget()
        asyncio.create_task(self.game_controller.reset(human_colors))
        self.game_screen.pack(fill=tk.BOTH, expand=True)
        self.current_screen = self.game_screen

    def create_menu(self):
        menu_bar = tk.Menu(self)
        self.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Exit", command=self.quit)

    def open_file(self):
        file_path = filedialog.askopenfilename()
        print("Selected file:", file_path)


class TitleScreen(tk.Frame):
    def __init__(self, master, switch_screen_callback):
        super().__init__(master)

        # Load and display the background image
        logo = Image.open("icons/Logo.jpg").resize((614, 608))
        image = ImageTk.PhotoImage(logo)
        label = tk.Label(self, image=image)
        label.image = image  # avoid garbage collection
        label.pack(fill=tk.BOTH, expand=True)

        # Create a custom style for the buttons
        style = ttk.Style()
        style.configure(
            "TButton",
            foreground="#545454",
            background="white",
            font=("Helvetica", 14),
            padding=8,
            relief=tk.FLAT,
        )

        # Create a custom style for the buttons
        style = ttk.Style()
        style.theme_use("clam")  # put the theme name here, that you want to use
        style.configure(
            "TButton",
            foreground=DARK_COLOR,
            background=GREY_COLOR,
            font=("Cascadia Mono", 12),
            relief=tk.FLAT,
            anchor=tk.CENTER,
        )

        # Create the "Go to Title Screen" button
        self.white_button = ttk.Button(
            self,
            text="Play as White",
            command=lambda: switch_screen_callback([chess.WHITE]),
            style="TButton",
            compound=tk.CENTER,
        )

        self.black_button = ttk.Button(
            self,
            text="Play as Black",
            command=lambda: switch_screen_callback([chess.BLACK]),
            style="TButton",
            compound=tk.CENTER,
        )

        self.both_button = ttk.Button(
            self,
            text="Play as Both ",
            command=lambda: switch_screen_callback([chess.WHITE, chess.BLACK]),
            style="TButton",
            compound=tk.CENTER,
        )

        self.white_button.place(relx=0.5, rely=0.75, anchor=tk.CENTER)
        self.black_button.place(relx=0.5, rely=0.83, anchor=tk.CENTER)
        self.both_button.place(relx=0.5, rely=0.91, anchor=tk.CENTER)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    app = Application(loop)
    asyncio.run(app.show_async())
