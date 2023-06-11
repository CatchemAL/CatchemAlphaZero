from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import product
from typing import cast

import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from ..games import Game
from ..solvers import Solver
from ..states import State, TMove
from ..states.state import Policy, TemperatureSchedule
from .alpha_zero_mcts import AlphaZeroMcts
from .alpha_zero_parameters import AZArenaParameters, AZMctsParameters, AZTrainingParameters
from .alpha_zero_training_models import ParallelGame, RecordedAction
from .network import NeuralNetwork, TrainingData


class AlphaZero:
    def __init__(self, neural_net: NeuralNetwork) -> None:
        self.neural_net = neural_net

    @property
    def game(self) -> Game:
        return self.neural_net.game

    def as_solver(self, num_mcts_sims: int | AZMctsParameters) -> Solver:
        return AlphaZeroSolver(self, num_mcts_sims)

    def policy(self, state: State[TMove], num_mcts_sims: int | AZMctsParameters) -> Policy:
        self.neural_net.set_to_eval()
        mcts = self._get_mcts(num_mcts_sims)
        return mcts.policy(state)

    def policies(
        self, states: list[State[TMove]], num_mcts_sims: int | AZMctsParameters
    ) -> list[Policy]:
        self.neural_net.set_to_eval()
        mcts = self._get_mcts(num_mcts_sims)
        return mcts.policies(states)

    def raw_policy(self, state: State[TMove]) -> tuple[np.ndarray, float]:
        self.neural_net.set_to_eval()
        policy, value = self.neural_net.predict(state)
        return policy, value

    def select_move(
        self,
        state: State[TMove],
        num_mcts_sims: int | AZMctsParameters,
        temperature_schedule: TemperatureSchedule | None = None,
    ) -> TMove:
        schedule = temperature_schedule or TemperatureSchedule.competitive()
        temperature = schedule.get_temperature(state.move_count)
        policy = self.policy(state, num_mcts_sims)
        move = policy.select_move(temperature)
        return move

    def select_moves(
        self,
        states: list[State[TMove]],
        num_mcts_sims: int | AZMctsParameters,
        temperature_schedule: TemperatureSchedule | None = None,
    ) -> list[TMove]:
        schedule = temperature_schedule or TemperatureSchedule.competitive()
        temperatures = [schedule.get_temperature(state.move_count) for state in states]
        policies = self.policies(states, num_mcts_sims)
        moves = [policy.select_move(temperature) for policy, temperature in zip(policies, temperatures)]
        return moves

    def self_learn(
        self, training_params: AZTrainingParameters, initial_state: str | list[int] | None = None
    ) -> None:
        num_rounds = training_params.games_per_generation // training_params.num_parallel

        mcts = self._get_mcts(training_params.mcts_parameters)

        for _ in trange(training_params.num_generations, desc="Generations"):
            training_set: list[TrainingData] = []

            for _ in trange(num_rounds, desc="- Self-play", leave=False):
                # training_set += self.self_play(mcts, training_params.temperature, initial_state)
                training_set += self.self_play_parallel(
                    mcts,
                    training_params.num_parallel,
                    training_params.temperature,
                    initial_state,
                )

            extended_training_set = self._exploit_symmetries(training_set)

            broken = """

            neural_net_after = deepcopy(self.neural_net)
            neural_net_after.train(extended_training_set, num_epochs, minibatch_size)
            neural_net_after.generation += 1

            # Battle the newly trained net in the arena
            arena = Arena(self.game, **training_params.arena_parameters.__dict__)
            alpha_zero_after = AlphaZero(neural_net_after)
            arena_result = arena.battle(self, alpha_zero_after)
            if arena_result.is_improvement or not arena_result.must_improve:
                self.neural_net = neural_net_after
                self.neural_net.save()
            """

            # Train against the newly generated games
            num_epochs, minibatch_size = training_params.num_epochs, training_params.minibatch_size
            self.neural_net.save_training_data(extended_training_set)
            neural_net_before = deepcopy(self.neural_net)
            self.neural_net.train(extended_training_set, num_epochs, minibatch_size)
            self.neural_net.generation += 1

            # Battle
            if True:
                arena = Arena(self.game, **training_params.arena_parameters.__dict__)
                alpha_zero_before = AlphaZero(neural_net_before)
                arena_result = arena.battle(alpha_zero_before, self)
                if arena_result.must_improve and not arena_result.is_improvement:
                    self.neural_net = neural_net_before

            self.neural_net.save()

    def self_play(
        self,
        mcts: AlphaZeroMcts,
        temperature_schedule: TemperatureSchedule,
        initial_state: str | list[int] | None = None,
    ) -> list[TrainingData]:
        state: State = self.game.initial_state(initial_state)
        status = state.status()

        self.neural_net.set_to_eval()
        recorded_actions: list[RecordedAction] = []

        while status.is_in_progress:
            state_before = state
            policy = mcts.policy(state_before)
            temperature = temperature_schedule.get_temperature(state_before.move_count)
            move = policy.select_move(temperature)
            state = state_before.play_move(move)
            recorded_action = RecordedAction(state_before, policy.encoded_policy, move, state)
            recorded_actions.append(recorded_action)
            status = state.status()

        training_set: list[TrainingData] = []
        for ph in recorded_actions:
            encoded_state = ph.state_before.to_feature()
            perspective = ph.state_after.played_by
            outcome = status.outcome(perspective)
            training_data = TrainingData(encoded_state, ph.policy, outcome)
            training_set.append(training_data)

        return training_set

    def self_play_parallel(
        self,
        solver: AlphaZeroMcts,
        num_parallel: int,
        temperature_schedule: TemperatureSchedule,
        initial_state: str | list[int] | None = None,
    ) -> list[TrainingData]:
        initial_states: list[State] = self.game.generate_states(2, num_parallel, initial_state)
        parallel_games = [ParallelGame(s) for s in initial_states if s.status().is_in_progress]
        in_progress_games = list(parallel_games)

        self.neural_net.set_to_eval()

        progress_bar = tqdm.tqdm(total=float("inf"), unit=" -- count:", leave=False)
        while in_progress_games:
            states = [pg.latest_state for pg in in_progress_games]
            policies = solver.policies(states)

            for policy, pg in zip(policies, in_progress_games):
                policy, pg = cast(Policy, policy), cast(ParallelGame, pg)
                state_before = pg.latest_state
                temperature = temperature_schedule.get_temperature(state_before.move_count)
                move = policy.select_move(temperature)
                state = state_before.play_move(move)
                recorded_action = RecordedAction(state_before, policy.encoded_policy, move, state)
                pg.recorded_actions.append(recorded_action)

            in_progress_games = [g for g in in_progress_games if g.latest_state.status().is_in_progress]
            progress_bar.update(1)
        progress_bar.close()

        training_set: list[TrainingData] = []
        for pg in parallel_games:
            terminal_status = pg.latest_state.status()
            policy_history = pg.recorded_actions
            for ph in policy_history:
                encoded_state = ph.state_before.to_feature()
                perspective = ph.state_after.played_by
                outcome = terminal_status.outcome(perspective)
                training_data = TrainingData(encoded_state, ph.policy, outcome)
                training_set.append(training_data)

        return training_set

    def explore_hyperparameters(self, generation: int, arena_parameters: AZArenaParameters) -> None:
        hps = {
            "minibatch_size": [256, 512, 1024, 2048, 4096],
            "num_epochs": [2, 4, 6, 8, 10, 12],
            "learning_rate": [0.01, 0.001, 0.000_1, 0.000_01],
        }

        writer = SummaryWriter(log_dir=f"runs/{self.game.fullname}_Gen{generation:03d}")
        arena = Arena(self.game, **arena_parameters.__dict__)
        training_set = self.neural_net.load_training_data(generation)

        all_hyper_params = list(product(hps["num_epochs"], hps["minibatch_size"]))
        for num_epochs, minibatch_size in trange(all_hyper_params, desc="Hyperparameter Search"):
            neural_net_after = deepcopy(self.neural_net)

            # Train the network with the current hyperparameters
            neural_net_after.train(training_set, num_epochs, minibatch_size, log_progress=False)
            alpha_zero_after = AlphaZero(neural_net_after)
            arena_result = arena.battle(self, alpha_zero_after, log_progress=False)

            hparams = {"minibatch_size": minibatch_size, "num_epochs": num_epochs}
            metrics = {"win_ratio": arena_result.win_ratio}
            writer.add_hparams(hparams, metrics)

        writer.close()

    def _get_mcts(self, num_mcts_sims: int | AZMctsParameters) -> AlphaZeroMcts:
        if isinstance(num_mcts_sims, int):
            default_params = AZMctsParameters.defaults(self.game.fullname)
            value_by_prop = asdict(default_params)
            value_by_prop["num_mcts_sims"] = num_mcts_sims
        else:
            value_by_prop = asdict(num_mcts_sims)

        return AlphaZeroMcts(self.neural_net, **value_by_prop)

    def _exploit_symmetries(self, training_set: list[TrainingData]) -> list[TrainingData]:
        def all_sets(data: TrainingData) -> list[TrainingData]:
            symmetries = self.game.symmetries(data.encoded_state, data.policy)
            return (TrainingData(state, pol, data.outcome) for state, pol in symmetries)

        return [sym_data for training_data in training_set for sym_data in all_sets(training_data)]

    def _get_random_state(
        self, num_moves: int, probability: float, initial_state: str | list[int] | None = None
    ) -> State:
        state: State = self.game.initial_state(initial_state)
        if np.random.rand() > probability:
            return state

        for _ in range(num_moves):
            status = state.status()
            if status.is_in_progress:
                move = np.random.choice(status.legal_moves)
                state.play_move(move)

        return state


@dataclass(slots=True)
class AlphaZeroSolver(Solver):
    alpha_zero: AlphaZero
    num_mcts_sims: int

    def solve(self, state: State[TMove]) -> TMove:
        return self.alpha_zero.select_move(state, self.num_mcts_sims)


class Arena:
    def __init__(
        self,
        game: Game,
        *,
        num_games: int,
        num_mcts_sims: int,
        required_win_ratio: float,
        temperature_schedule: TemperatureSchedule,
    ) -> None:
        self.game = game
        self.num_games = num_games
        self.num_mcts_sims = num_mcts_sims
        self.required_win_ratio = required_win_ratio
        self.temperature_schedule = temperature_schedule

    def battle(
        self, current: AlphaZero, challenger: AlphaZero, log_progress: bool = True
    ) -> ArenaResult:
        games_to_play = self.num_games // 2

        num_wins1, num_draws1, num_losses1 = self.play_games(current, challenger, games_to_play)
        num_wins2, num_draws2, num_losses2 = self.play_games(challenger, current, games_to_play)

        arena_result = ArenaResult(
            challenger.neural_net.generation,
            num_wins2 + num_losses1,
            num_draws1 + num_draws2,
            num_losses2 + num_wins1,
            self.required_win_ratio,
        )

        if log_progress:
            self._log_result(arena_result)

        return arena_result

    def play_games(
        self, player1: AlphaZero, player2: AlphaZero, games_to_play: int
    ) -> tuple[float, float, float]:
        initial_states: list[State] = self.game.generate_states(2, games_to_play)

        active_states = [state for state in initial_states if state.status().is_in_progress]
        finished_states: list[State] = []

        player, opponent = player1, player2
        while active_states:
            moves = player.select_moves(active_states, self.num_mcts_sims, self.temperature_schedule)
            for i, move in enumerate(moves):
                active_states[i] = active_states[i].play_move(move)

            player, opponent = opponent, player

            statuses = [state.status().is_in_progress for state in active_states]
            finished_states += [s for s, in_progress in zip(active_states, statuses) if not in_progress]
            active_states = [s for s, in_progress in zip(active_states, statuses) if in_progress]

        outcomes = [state.status().outcome(1) for state in finished_states]
        num_wins = len([outcome for outcome in outcomes if int(outcome) == 1])
        num_draws = len([outcome for outcome in outcomes if int(outcome) == 0])
        num_losses = games_to_play - num_wins - num_draws
        return num_wins, num_draws, num_losses

    def _log_result(self, arena_result: ArenaResult) -> None:
        writer = SummaryWriter(log_dir=f"runs/{self.game.fullname}")
        gen = arena_result.generation

        # Logging metrics to TensorBoard
        writer.add_scalar(f"Arena/Win Ratio", arena_result.win_ratio, gen)
        writer.add_scalar(f"Arena/Total Wins", arena_result.num_wins, gen)
        writer.add_scalar(f"Arena/Total Draws", arena_result.num_draws, gen)
        writer.add_scalar(f"Arena/Total Losses", arena_result.num_losses, gen)
        writer.close()


@dataclass
class InitialStateGenerator:
    initial_state: State
    depth: int

    def generate(self):
        pass


@dataclass(slots=True)
class ArenaResult:
    generation: int
    num_wins: float
    num_draws: float
    num_losses: float
    required_ratio_cutoff: float
    must_improve: bool = False

    @property
    def win_ratio(self) -> bool:
        return (self.num_wins + 0.5 * self.num_draws) / self.num_games

    @property
    def num_games(self) -> bool:
        return self.num_wins + self.num_draws + self.num_losses

    @property
    def is_improvement(self) -> bool:
        return self.win_ratio > self.required_ratio_cutoff
