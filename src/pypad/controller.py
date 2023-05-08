from dataclasses import dataclass
from typing import Generic

from .games.state import StateFactory, StateView, TMove, TState
from .solvers.solver import Solver


@dataclass
class Controller(Generic[TState, TMove]):
    player1: Solver
    player2: Solver
    state_factory: StateFactory[TState]
    view: StateView[TState]

    def run(self, init: str) -> None:
        state = self.state_factory.load_initial_state(init)

        player, opponent = self.player1, self.player2
        while list(state.legal_moves()):
            move = player.solve(state)
            state.play_move(move)
            self.view.display(state)
            player, opponent = opponent, player

        if state.is_won():
            print(f"Player {state.played_by}  wins!")
        else:
            print("It's a draw!")
