from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

from pypad.views.chess_detail import ChessScreen, DARK_COLOR

GREY_COLOR = "#BEC7C6"


class ChessScreenController:
    def __init__(self, model, view: ChessScreen):
        self.model = model
        self.view = view
        self.view.set_button_command(self.go_to_screen2)


class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("CatchemAlphaZero")
        self.resizable(False, False)
        self.iconbitmap("icons/tiny_chess.ico")

        self.current_screen = None
        self.title_screen = TitleScreen(self, self.switch_to_game_screen)
        self.game_screen = ChessScreen(self, self.switch_to_title_screen)

        self.switch_to_title_screen()
        self.create_menu()

    def switch_to_title_screen(self):
        if self.current_screen:
            self.current_screen.pack_forget()
        self.title_screen.pack(fill=tk.BOTH, expand=True)
        self.current_screen = self.title_screen

    def switch_to_game_screen(self):
        if self.current_screen:
            self.current_screen.pack_forget()
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
        white_button = ttk.Button(
            self,
            text="Play as White",
            command=switch_screen_callback,
            style="TButton",
            compound=tk.CENTER,
        )

        black_button = ttk.Button(
            self,
            text="Play as Black",
            command=switch_screen_callback,
            style="TButton",
            compound=tk.CENTER,
        )

        both_button = ttk.Button(
            self,
            text="Play as Both ",
            command=switch_screen_callback,
            style="TButton",
            compound=tk.CENTER,
        )

        white_button.place(relx=0.5, rely=0.75, anchor=tk.CENTER)
        black_button.place(relx=0.5, rely=0.83, anchor=tk.CENTER)
        both_button.place(relx=0.5, rely=0.91, anchor=tk.CENTER)


if __name__ == "__main__":
    app = Application()
    app.mainloop()
