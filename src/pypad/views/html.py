from abc import ABC, abstractmethod
from typing import Generic, Tuple

import numpy as np

from pypad.states import ConnectXState, TicTacToeState

from ..solvers.mcts import Node
from ..states import TState


class HtmlBuilder(ABC, Generic[TState]):
    @abstractmethod
    def build_html(self, state: TState) -> str:
        ...

    @abstractmethod
    def build_tiny_html(self, state: TState) -> str:
        ...

    @abstractmethod
    def display(self, state: TState) -> None:
        ...

    def _html(self, grid: np.ndarray, is_col_ordinal: bool) -> str:
        tiny_html = self._tiny_html(grid, True, is_col_ordinal)

        html_table = f"""
        <table class="ttt-board" style="margin:auto;text-align:center;float:left">
            <tbody>
                {tiny_html}
            </tbody>
        </table>"""

        html_with_css = (
            """
<!DOCTYPE html>
<html>
    <head>
        <title></title>
        <meta charset="UTF-8">
        <style>
            .ttt-board { border-spacing: 0; border-collapse: collapse; }
            .ttt-board th { padding: .5em; }
            .ttt-board th + th { border-bottom: 1px solid #000; }
            .ttt-board th:first-child,
            .ttt-board td:last-child { border-right: 1px solid #000; }
            .ttt-board tr:last-child td { border-bottom: 1px solid; }
            .ttt-board th:empty { border: none; }
            .ttt-board td { width: 1.5em; height: 1.5em; text-align: center; font-size: 18px; line-height: 0;}
        </style>
    </head>
    <body>"""
            + html_table
            + """
    </body>
</html>
"""
        )
        return html_with_css

    def _tiny_html(
        self, grid: np.ndarray, include_headers: bool = False, is_col_ordinal: bool = False
    ) -> str:
        rows, cols = grid.shape

        def piece(i: int, j: int) -> str:
            mark = grid[i, j]
            match mark:
                case 1:
                    return r'<font POINT-SIZE="12">⭕</font>'
                case 2:
                    return r'<font POINT-SIZE="12">✖️</font>'
                case _:
                    return ""

        table = ""

        if include_headers:
            table += "        <tr>\n"
            table += "            <th></th>\n"
            for col in range(cols):
                letter = col + 1 if is_col_ordinal else chr(ord("a") + col)
                table += f"            <th><center>{letter}</center></th>\n"
            table += "        </tr>\n"

        for row in range(rows):
            table += "        <tr>\n"
            if include_headers:
                table += f"            <th>{rows - row}</th>\n"
            for col in range(cols):
                mark = piece(row, col)
                color = "#FFFFFF" if (row + col) % 2 == 1 else "#EBEBEB"
                table += f'            <td height="20" width="20" bgcolor="{color}">{mark}</td>\n'
            table += "        </tr>\n"
        return table


class MctsNodeHtmlBuilder:
    def build_tiny_html(self, node: Node, state: TState) -> str:
        _, cols = state.shape

        win_label, ucb_label = MctsNodeHtmlBuilder._get_labels(node)

        tiny = state.html(True)
        html = f"""<<table cellborder="0" cellspacing="0" border="0">
            {tiny}
            <tr>
                <td height="5" colspan="{cols}" align="center"></td>
            </tr>
            <tr>
                <td colspan="{cols}" align="center"><font face="Helvetica" color="#424242" POINT-SIZE="11">{win_label}</font></td>
            </tr>
            <tr>
                <td colspan="{cols}" align="center"><font face="Helvetica" color="#424242" POINT-SIZE="11">{ucb_label}</font></td>
            </tr>
        </table>>"""

        return html

    @staticmethod
    def _get_labels(node: Node) -> Tuple[str, str]:
        win_label = f"N={node.visit_count}, W={node.wins}"
        ucb_label = f"UCB={node.ucb():.4}" if node.parent else "root"
        return win_label, ucb_label


class TicTacToeHtmlBuilder(HtmlBuilder[TicTacToeState]):
    def display(self, state: TicTacToeState) -> None:
        from IPython.display import HTML

        html_string = self.build_html(state)
        return HTML(html_string)

    def build_tiny_html(self, state: TicTacToeState) -> str:
        grid = state.to_grid()
        return self._tiny_html(grid)

    def build_html(self, state: TicTacToeState) -> str:
        grid = state.to_grid()
        return self._html(grid, False)


class ConnectXHtmlBuilder(HtmlBuilder[ConnectXState]):
    def display(self, state: ConnectXState) -> None:
        from IPython.display import HTML

        html_string = self.build_html(state)
        return HTML(html_string)

    def build_tiny_html(self, state: ConnectXState) -> str:
        grid = state.to_grid()
        return self._tiny_html(grid)

    def build_html(self, state: ConnectXState) -> str:
        grid = state.to_grid()
        return self._html(grid, True)


class ChessHtmlView:
    def __init__(self):
        self.path_by_piece = {
            "wp": '<img src="icons/chess-pawn-regular.svg" width="29" style="display: block; margin: auto;">',
            "wr": '<img src="icons/chess-rook-regular.svg" width="42" style="display: block; margin: auto;">',
            "wb": '<img src="icons/chess-bishop-regular.svg" width="31" style="display: block; margin: auto;">',
            "wn": '<img src="icons/chess-knight-regular.svg" width="41" style="display: block; margin: auto;">',
            "wk": '<img src="icons/chess-king-regular.svg" width="46" style="display: block; margin: auto;">',
            "wq": '<img src="icons/chess-queen-regular.svg" height="45" style="display: block; margin: auto;">',
            "bp": '<img src="icons/chess-pawn-solid.svg" width="30" style="display: block; margin: auto;">',
            "br": '<img src="icons/chess-rook-solid.svg" width="40" style="display: block; margin: auto;">',
            "bb": '<img src="icons/chess-bishop-solid.svg" width="30" style="display: block; margin: auto;">',
            "bn": '<img src="icons/chess-knight-solid.svg" width="41" style="display: block; margin: auto;">',
            "bk": '<img src="icons/chess-king-solid.svg" width="44" style="display: block; margin: auto;">',
            "bq": '<img src="icons/chess-queen-solid.svg" height="45" style="display: block; margin: auto;">',
        }

    def display(self, piece_by_position) -> None:
        from IPython.display import HTML

        html_string = self.build_html(piece_by_position)
        return HTML(html_string)

    def build_html(self, piece_by_position) -> None:
        def piece(pos: str) -> str:
            if not pos or str.isspace(pos):
                return None

            col = pos[0]
            row = int(pos[1])

            return piece_by_position[8 - row][ord(col) - ord("A")]

        def img(pos: str) -> str:
            pc = piece(pos)

            if not pc or str.isspace(pc):
                return ""

            return self.path_by_piece[pc]

        grid_html = f"""
        <table class="chess-board" style="margin:auto;text-align:center;">
            <tbody>
                <tr>
                    <th></th>
                    <th><center>a</center></th>
                    <th><center>b</center></th>
                    <th><center>c</center></th>
                    <th><center>d</center></th>
                    <th><center>e</center></th>
                    <th><center>f</center></th>
                    <th><center>g</center></th>
                    <th><center>h</center></th>
                </tr>
                <tr>
                    <th>8</th>
                    <td class="dark" >{img('A8')}</td>
                    <td class="light">{img('B8')}</td>
                    <td class="dark" >{img('C8')}</td>
                    <td class="light">{img('D8')}</td>
                    <td class="dark" >{img('E8')}</td>
                    <td class="light">{img('F8')}</td>
                    <td class="dark" >{img('G8')}</td>
                    <td class="light">{img('H8')}</td>
                </tr>
                <tr>
                    <th>7</th>
                    <td class="light">{img('A7')}</td>
                    <td class="dark" >{img('B7')}</td>
                    <td class="light">{img('C7')}</td>
                    <td class="dark" >{img('D7')}</td>
                    <td class="light">{img('E7')}</td>
                    <td class="dark" >{img('F7')}</td>
                    <td class="light">{img('G7')}</td>
                    <td class="dark" >{img('H7')}</td>
                </tr>
                <tr>
                    <th>6</th>
                    <td class="dark" >{img('A6')}</td>
                    <td class="light">{img('B6')}</td>
                    <td class="dark" >{img('C6')}</td>
                    <td class="light">{img('D6')}</td>
                    <td class="dark" >{img('E6')}</td>
                    <td class="light">{img('F6')}</td>
                    <td class="dark" >{img('G6')}</td>
                    <td class="light">{img('H6')}</td>
                </tr>
                <tr>
                    <th>5</th>
                    <td class="light">{img('A5')}</td>
                    <td class="dark" >{img('B5')}</td>
                    <td class="light">{img('C5')}</td>
                    <td class="dark" >{img('D5')}</td>
                    <td class="light">{img('E5')}</td>
                    <td class="dark" >{img('F5')}</td>
                    <td class="light">{img('G5')}</td>
                    <td class="dark" >{img('H5')}</td>
                </tr>              
                <tr>
                    <th>4</th>
                    <td class="dark" >{img('A4')}</td>
                    <td class="light">{img('B4')}</td>
                    <td class="dark" >{img('C4')}</td>
                    <td class="light">{img('D4')}</td>
                    <td class="dark" >{img('E4')}</td>
                    <td class="light">{img('F4')}</td>
                    <td class="dark" >{img('G4')}</td>
                    <td class="light">{img('H4')}</td>
                </tr>
                <tr>
                    <th>3</th>
                    <td class="light">{img('A3')}</td>
                    <td class="dark" >{img('B3')}</td>
                    <td class="light">{img('C3')}</td>
                    <td class="dark" >{img('D3')}</td>
                    <td class="light">{img('E3')}</td>
                    <td class="dark" >{img('F3')}</td>
                    <td class="light">{img('G3')}</td>
                    <td class="dark" >{img('H3')}</td>
                </tr> 
                <tr>
                    <th>2</th>
                    <td class="dark" >{img('A2')}</td>
                    <td class="light">{img('B2')}</td>
                    <td class="dark" >{img('C2')}</td>
                    <td class="light">{img('D2')}</td>
                    <td class="dark" >{img('E2')}</td>
                    <td class="light">{img('F2')}</td>
                    <td class="dark" >{img('G2')}</td>
                    <td class="light">{img('H2')}</td>
                </tr>
                <tr>
                    <th>1</th>
                    <td class="light">{img('A1')}</td>
                    <td class="dark" >{img('B1')}</td>
                    <td class="light">{img('C1')}</td>
                    <td class="dark" >{img('D1')}</td>
                    <td class="light">{img('E1')}</td>
                    <td class="dark" >{img('F1')}</td>
                    <td class="light">{img('G1')}</td>
                    <td class="dark" >{img('H1')}</td>
                </tr>
            </tbody>
        </table>"""

        html_string = (
            """
<!DOCTYPE html>
<html>
    <head>
        <title></title>
        <meta charset="UTF-8">
        <style>
            .chess-board { border-spacing: 0; border-collapse: collapse; }
            .chess-board th { padding: .5em; }
            .chess-board th + th { border-bottom: 1px solid #000; }
            .chess-board th:first-child,
            .chess-board td:last-child { border-right: 1px solid #000; }
            .chess-board tr:last-child td { border-bottom: 1px solid; }
            .chess-board th:empty { border: none; }
            .chess-board td { width: 1.5em; height: 1.5em; text-align: center; font-size: 32px; line-height: 0;}
            .chess-board .light { background: #346B51; }
            .chess-board .dark { background: #EDF9EB; }
        </style>
    </head>
    <body>"""
            + grid_html
            + """
    </body>
</html>
"""
        )
        return html_string
