from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ..states import ConnectXState, State, TicTacToeState
from ..states.state import Policy


@dataclass
class PoliciedState:
    state: State
    policy_grid: NDArray

    def _repr_html_(self) -> str:
        return self.state.html(self.policy_grid, False)


def show_tictactoe_policy(
    self: TicTacToeState, policy: Policy, include_chart: bool = True
) -> PoliciedState:
    encoded_policy = policy.encoded_policy
    q_value = policy.value

    if include_chart:
        _, ax = plt.subplots(figsize=(5, 2))
        ax.spines["top"].set_edgecolor("white")
        ax.spines["right"].set_edgecolor("white")
        ax.spines["bottom"].set_edgecolor("black")
        ax.spines["left"].set_edgecolor("white")
        ax.set_yticks([])
        ax.set_xlabel(f"v ∈ (-1, +1) = {q_value:.2}")

        plt.bar(["a3", "b3", "c3", "a2", "b2", "c2", "a1", "b1", "c1"], encoded_policy, axes=ax)

    policy_grid = encoded_policy.reshape((self.rows, self.cols))
    return PoliciedState(self, policy_grid)


def show_connectx_policy(
    self: ConnectXState, policy: Policy, include_chart: bool = True
) -> PoliciedState:
    encoded_policy = policy.encoded_policy
    q_value = policy.value

    if include_chart:
        _, ax = plt.subplots(figsize=(4.34, 2))
        plt.bar(range(1, self.cols + 1), encoded_policy, axes=ax)

        ax.spines["top"].set_edgecolor("white")
        ax.spines["right"].set_edgecolor("white")
        ax.spines["bottom"].set_edgecolor("black")
        ax.spines["left"].set_edgecolor("white")
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel(f"v ∈ (-1, +1) = {q_value:.2}")

    policy_grid = np.tile(encoded_policy, (self.rows, 1))
    return PoliciedState(self, policy_grid)


ConnectXState.show_policy = show_connectx_policy
TicTacToeState.show_policy = show_tictactoe_policy
