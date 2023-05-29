class BitboardUtil:
    def __init__(self, rows: int, cols: int):
        self.rows: int = rows
        self.cols: int = cols
        self.num_slots: int = rows * cols
        self.BOTTOM_ROW: int = self.get_bottom_row()
        self.BOARD_MASK: int = self.get_board_mask()

    def get_bottom_row(self) -> int:
        x = 1
        for _ in range(self.cols - 1):
            x |= x << self.rows
        return x

    def get_col_mask(self, col: int) -> int:
        first_col = (1 << (self.rows - 1)) - 1
        return first_col << (self.rows * col)

    def move_to_col(self, move: int) -> int:
        for i in range(self.cols):
            col_mask = self.get_col_mask(i)
            if col_mask & move:
                return i

        raise ValueError(f"Move {move} not associated with any column.")

    def get_board_mask(self) -> int:
        x = self.BOTTOM_ROW << (self.rows - 1)
        return x - self.BOTTOM_ROW

    def move_order(self) -> list[int]:
        order: list[int] = [0] * self.cols

        for i in range(self.cols):
            if i % 2 == 0:
                order[i] = (self.cols - i - 1) // 2
            else:
                order[i] = (self.cols + i) // 2

        return order

    def __repr__(self) -> str:
        return f"BitboardUtil(rows={self.rows}, cols={self.cols})"
