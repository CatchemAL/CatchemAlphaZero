from dataclasses import dataclass

from ..games import Game
from ..states import State, Status
from .alpha_zero import AlphaZero


@dataclass
class Arena:
    game: Game
    num_games_to_play: int
    num_mcts_sim: int
    temperature: float

    def compete(self, current: AlphaZero, challenger: AlphaZero) -> float:
        games_to_play = self.num_games_to_play // 2

        num_wins1 = self.play_games(current, challenger, games_to_play)
        num_wins2 = self.play_games(challenger, current, games_to_play)

        total_wins = num_wins2 + (games_to_play - num_wins1)
        win_pct = total_wins / (2 * games_to_play)
        return win_pct

    def play_games(self, player1: AlphaZero, player2: AlphaZero, games_to_play: int) -> float:
        initial_states: list[State] = [self.game.initial_state() for _ in range(games_to_play)]
        states = list(initial_states)

        player, opponent = player1, player2
        while states:
            policies, _ = player.policies(states, self.num_mcts_sim)
            for state, policy in zip(states, policies):
                move = state.select_move(policy, self.temperature)
                state.play_move(move)

            player, opponent = opponent, player

            states = [state for state in states if state.status().is_in_progress]

        outcomes = [state.status().outcome(1) for state in states]
        return sum(outcomes)
