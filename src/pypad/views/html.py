from abc import ABC, abstractmethod
from typing import Generic

import numpy as np

from ..solvers.mcts import Node
from ..states import ConnectXState, TicTacToeState, TState
from .plot import heatmap_checker_colors, light_checker_colors


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

    def _html(self, grid: np.ndarray, policy: np.ndarray | None, has_numeric_cols: bool) -> str:
        tiny_html = self._tiny_html(
            grid,
            heatmap_values=policy,
            include_headers=True,
            has_numeric_cols=has_numeric_cols,
        )

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
        self,
        grid: np.ndarray,
        heatmap_values: np.ndarray | None = None,
        include_headers: bool = False,
        has_numeric_cols: bool = False,
    ) -> str:
        rows, cols = grid.shape

        pieces = np.vectorize(HtmlBuilder.piece)(grid)

        if heatmap_values is not None:
            colors = heatmap_checker_colors(heatmap_values)
        else:
            colors = light_checker_colors(grid.shape)

        table = ""

        if include_headers:
            table += "<tr>\n"
            table += "<th></th>\n"
            for col in range(cols):
                letter = col + 1 if has_numeric_cols else chr(ord("a") + col)
                table += f"<th><center>{letter}</center></th>\n"
            table += "</tr>\n"

        for row in range(rows):
            table += "<tr>\n"
            if include_headers:
                table += f"<th>{rows - row}</th>\n"
            for col in range(cols):
                mark = pieces[row, col]
                color = colors[row, col]
                table += f'<td height="20" width="20" bgcolor="{color}">{mark}</td>\n'
            table += "</tr>\n"
        return table

    @staticmethod
    def piece(mark) -> str:
        match mark:
            case 1:
                return r'<font POINT-SIZE="12">⭕</font>'
            case 2:
                return r'<font POINT-SIZE="12">✖️</font>'
            case _:
                return ""


class MctsNodeHtmlBuilder:
    def build_tiny_html(self, node: Node, state: TState) -> str:
        _, cols = state.shape

        win_label, ucb_label = MctsNodeHtmlBuilder._get_labels(node)

        tiny = state.html(is_tiny_repr=True)
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
    def _get_labels(node: Node) -> tuple[str, str]:
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

    def build_html(self, state: TicTacToeState, policy: np.ndarray | None = None) -> str:
        grid = state.to_grid()
        return self._html(grid, policy=policy, has_numeric_cols=False)


class ConnectXHtmlBuilder(HtmlBuilder[ConnectXState]):
    def display(self, state: ConnectXState) -> None:
        from IPython.display import HTML

        html_string = self.build_html(state)
        return HTML(html_string)

    def build_tiny_html(self, state: ConnectXState) -> str:
        grid = state.to_grid()
        return self._tiny_html(grid)

    def build_html(self, state: ConnectXState, policy: np.ndarray | None = None) -> str:
        grid = state.to_grid()
        return self._html(grid, policy=policy, has_numeric_cols=True)
