import matplotlib.pyplot as plt
import numpy as np


def create_checkered_array(n: int) -> np.ndarray:
    arr = np.ones((n, n), dtype=int)
    arr[1::2, ::2] = 0
    arr[::2, 1::2] = 0
    return arr


def checker_pattern_color(i, j):
    is_white = (i + j) % 2 == 0
    return "#FFFFFF" if is_white else "#000000"


def plot_state(planes: np.ndarray, figsize: tuple[int, int], linewidth: float = 1) -> None:
    _, rows, cols = planes.shape
    _, ax = plt.subplots(figsize=figsize)
    _ = ax.imshow(planes.transpose(1, 2, 0))

    ax.set_xticks(np.arange(-0.5, rows, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cols, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=linewidth)

    ax.spines["top"].set_edgecolor("white")
    ax.spines["right"].set_edgecolor("white")
    ax.spines["bottom"].set_edgecolor("white")
    ax.spines["left"].set_edgecolor("white")

    ax.tick_params(axis="both", which="both", bottom=False, left=False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title("Player   =   Red\nOpponent = Green", loc="right", fontname="Monospace", fontsize=9)


def plot_chess_slice(
    planes: np.ndarray, slice: int, figsize: tuple[int, int], linewidth: float = 2.0
) -> None:
    _, rows, cols = planes.shape
    plane = planes[slice, :, :]

    plane += 0.1 * create_checkered_array(rows)
    _, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap("inferno")
    _ = ax.imshow(plane, cmap=cmap, vmin=0, vmax=1.1)

    ax.set_xticks(np.arange(-0.5, rows, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cols, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=linewidth)

    ax.spines["top"].set_edgecolor("white")
    ax.spines["right"].set_edgecolor("white")
    ax.spines["bottom"].set_edgecolor("white")
    ax.spines["left"].set_edgecolor("white")

    ax.tick_params(axis="both", which="both", bottom=False, left=False)
    ax.set_xticks([])
    ax.set_yticks([])


def heatmap_color(value: float) -> str:
    cmap = plt.get_cmap("summer")

    normalized_value = 1 - max(0, min(1, value))
    colormap_index = normalized_value * (cmap.N - 1)
    rgb_value = cmap(int(colormap_index))

    return "#{:02x}{:02x}{:02x}".format(
        int(rgb_value[0] * 255),
        int(rgb_value[1] * 255),
        int(rgb_value[2] * 255),
    )


def combine_hex_colors(color1, color2, transparency2):
    rgb1 = (int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16))
    rgb2 = (int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16))

    combined_color = [int(rgb2[i] * transparency2 + rgb1[i] * (1 - transparency2)) for i in range(3)]
    return "#{:02x}{:02x}{:02x}".format(*combined_color[:3])


def heatmap_checker_colors(matrix: np.ndarray, include_checkers: bool = True) -> np.ndarray:
    if not include_checkers:
        return np.vectorize(heatmap_color)(matrix)

    heatmap_colors = np.vectorize(heatmap_color)(matrix)
    checker_pattern_colors = np.vectorize(checker_pattern_color)(*np.indices(matrix.shape))
    combined_colors = np.vectorize(combine_hex_colors)(heatmap_colors, checker_pattern_colors, 0.08)
    return combined_colors


def light_checker_colors(shape: tuple[int, int]) -> np.ndarray:
    def col(row, col) -> str:
        return "#FFFFFF" if (row + col) % 2 == 1 else "#EBEBEB"

    return np.vectorize(col)(*np.indices(shape))
