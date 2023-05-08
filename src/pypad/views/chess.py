import tkinter as tk


class ChessGUI:
    CELL_SIZE = 80

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Chess")

        self.player_turn = "W"
        self.selected_piece = None

        self.board = [
            ["BR", "BN", "BB", "BQ", "BK", "BB", "BN", "BR"],
            ["BP", "BP", "BP", "BP", "BP", "BP", "BP", "BP"],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["WP", "WP", "WP", "WP", "WP", "WP", "WP", "WP"],
            ["WR", "WN", "WB", "WQ", "WK", "WB", "WN", "WR"],
        ]

        # Store PhotoImage objects as instance variables
        self.images = {
            "WP": tk.PhotoImage(file=r"icons/white-pawn.png"),
            "WR": tk.PhotoImage(file=r"icons/white-rook.png"),
            "WN": tk.PhotoImage(file=r"icons/white-knight.png"),
            "WB": tk.PhotoImage(file=r"icons/white-bishop.png"),
            "WQ": tk.PhotoImage(file=r"icons/white-queen.png"),
            "WK": tk.PhotoImage(file=r"icons/white-king.png"),
            "BP": tk.PhotoImage(file=r"icons/black-pawn.png"),
            "BR": tk.PhotoImage(file=r"icons/black-rook.png"),
            "BN": tk.PhotoImage(file=r"icons/black-knight.png"),
            "BB": tk.PhotoImage(file=r"icons/black-bishop.png"),
            "BQ": tk.PhotoImage(file=r"icons/black-queen.png"),
            "BK": tk.PhotoImage(file=r"icons/black-king.png"),
        }

        # Create a canvas to draw the chess board and pieces
        self.canvas = tk.Canvas(self.window, width=700, height=700)
        self.canvas.pack()

        # Draw the chess board and pieces
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
        for i in range(8):
            for j in range(8):
                x1 = j * self.CELL_SIZE
                y1 = i * self.CELL_SIZE
                x2 = x1 + self.CELL_SIZE
                y2 = y1 + self.CELL_SIZE
                color = "#EDF9EB" if (i + j) % 2 == 0 else "#346B51"
                rect = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline="", tags="my_rectangle"
                )
                piece = self.board[i][j]

                if piece:
                    self.canvas.create_text(
                        x1 + self.CELL_SIZE // 2, y1 + self.CELL_SIZE // 2, text=piece
                    )

        self.draw_pieces()

    def draw_pieces(self):
        for i in range(8):
            for j in range(8):
                piece = self.board[j][i]
                if piece:
                    self.canvas.create_image(
                        i * self.CELL_SIZE + self.CELL_SIZE // 2,
                        j * self.CELL_SIZE + self.CELL_SIZE // 2,
                        image=self.images[piece],
                    )

    def on_click(self, event):
        x = event.x // self.CELL_SIZE
        y = event.y // self.CELL_SIZE
        piece = self.board[y][x]

        if not self.selected_piece:
            if piece and piece[0] == self.player_turn:
                self.selected_piece = (x, y)
                # Highlight the selected cell
                self.highlight_cell(x, y)
        else:
            if self.selected_piece == (x, y):
                self.selected_piece = None
                self.draw_board()  # redraw the board after each move
            elif self.move_piece(self.selected_piece, (x, y)):
                self.player_turn = "W" if self.player_turn == "B" else "B"
                self.selected_piece = None
                self.draw_board()  # redraw the board after each move

    def highlight_cell(self, x, y):
        x1 = x * self.CELL_SIZE
        y1 = y * self.CELL_SIZE
        x2 = x1 + self.CELL_SIZE
        y2 = y1 + self.CELL_SIZE
        highlighted_cell = self.canvas.create_rectangle(
            x1 + 2, y1 + 2, x2 - 2, y2 - 2, fill="#CEF261", outline="#CEF261", width=4, tag="highlight"
        )
        self.canvas.tag_lower(highlighted_cell, "all")
        self.canvas.tag_raise(highlighted_cell, "my_rectangle")

    def remove_highlight(self):
        self.canvas.delete("highlight")

    def move_piece(self, from_pos, to_pos):
        from_x, from_y = from_pos
        to_x, to_y = to_pos
        piece = self.board[from_y][from_x]

        if not piece or (to_x, to_y) == from_pos:
            return False

        # Check if the move is legal
        if piece[1] == "P":
            if piece[0] == "W":
                direction = -1
                start_row = 6
            else:
                direction = 1
                start_row = 1
            if from_y == start_row and to_y == start_row + 2 * direction:
                if self.board[from_y + direction][from_x] or self.board[from_y + 2 * direction][from_x]:
                    return False
            else:
                if from_y + direction != to_y or self.board[to_y][to_x]:
                    return False
            if abs(from_x - to_x) == 1 and abs(from_y - to_y) == 1:
                if not self.board[to_y][to_x]:
                    return False
            else:
                if from_x != to_x:
                    return False
            self.board[to_y][to_x] = piece
            self.board[from_y][from_x] = ""
        elif piece[1] == "R":
            if from_x != to_x and from_y != to_y:
                return False
            if from_x == to_x:
                start, end = min(from_y, to_y), max(from_y, to_y)
                if any(self.board[y][from_x] for y in range(start + 1, end)):
                    return False
            else:
                start, end = min(from_x, to_x), max(from_x, to_x)
                if any(self.board[from_y][x] for x in range(start + 1, end)):
                    return False
            self.board[to_y][to_x] = piece
            self.board[from_y][from_x] = ""
        # Add code for other pieces (bishop, knight, queen, king) here
        else:
            return False

        return True


# Create a new ChessGUI object and start the game
game = ChessGUI()
game.window.mainloop()
