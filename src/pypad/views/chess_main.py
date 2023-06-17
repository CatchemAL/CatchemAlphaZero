import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog
from pypad.views.chess_detail import ChessScreen


class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("CatchemAlphaZero")
        self.resizable(False, False)
        self.iconbitmap("icons/tiny_chess.ico")

        self.title_screen = TitleScreen(self, self.switch_to_game_screen)
        self.game_screen = ChessScreen(self, self.switch_to_title_screen)

        self.current_screen = None

        self.create_menu()
        self.switch_to_title_screen()

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
        logo = Image.open("icons/Logo.jpg").resize((512, 507))
        image = ImageTk.PhotoImage(logo)
        label = tk.Label(self, image=image)
        label.image = image
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

        white_button = ttk.Button(self, text="Play as White", command=switch_screen_callback)
        black_button = ttk.Button(self, text="Play as Black", command=switch_screen_callback)
        both_button = ttk.Button(self, text="Play as Both", command=switch_screen_callback)
        white_button.place(relx=0.5, rely=0.67, anchor=tk.CENTER)
        black_button.place(relx=0.5, rely=0.77, anchor=tk.CENTER)
        both_button.place(relx=0.5, rely=0.87, anchor=tk.CENTER)


class GameScreen(tk.Frame):
    def __init__(self, master, switch_screen_callback):
        super().__init__(master)

        # Create three buttons
        button1 = tk.Button(self, text="Button 1")
        button1.pack()

        button2 = tk.Button(self, text="Button 2")
        button2.pack()

        # Create the "Go to Title Screen" button
        title_button = tk.Button(self, text="Go to Title Screen", command=switch_screen_callback)
        title_button.pack()


class GameScreen(tk.Frame):
    def __init__(self, master, switch_screen_callback):
        super().__init__(master)

        # Create three buttons
        button1 = tk.Button(self, text="Button 1")
        button1.pack()

        button2 = tk.Button(self, text="Button 2")
        button2.pack()

        # Create the "Go to Title Screen" button
        title_button = tk.Button(self, text="Go to Title Screen", command=switch_screen_callback)
        title_button.pack()


if __name__ == "__main__":
    app = Application()
    app.mainloop()
