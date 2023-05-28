from ..states import State, TMove

import matplotlib.pyplot as plt

import numpy as np


def plot_state(state: State[TMove], figsize: tuple[int, int], linewidth: float = 1) -> None:
    rows, cols = state.shape
    planes = state.to_numpy()
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
