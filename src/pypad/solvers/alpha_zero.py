from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass, field

import numpy as np
from tqdm import trange

from ..games import Game
from ..states import State, TMove
from .alpha_zero_mcts import AlphaZeroMctsSolver
from .alpha_zero_parameters import AZMctsParameters, AZTrainingParameters
from .network import NeuralNetwork, TrainingData


@dataclass
class RecordedPolicy:
    state_before: State
    policy: np.ndarray
    move: int
    state_after: State


@dataclass
class ParallelGame:
    idx: int
    initial_state: State
    policy_history: list[RecordedPolicy] = field(default_factory=list)

    @property
    def latest_state(self) -> State:
        if self.policy_history:
            return self.policy_history[-1].state_after
        return self.initial_state


@dataclass
class AlphaZero:
    def __init__(self, neural_net: NeuralNetwork) -> None:
        self.neural_net = neural_net
        self.default_mcts_params = AZMctsParameters.defaults(self.game.fullname)

    @property
    def game(self) -> Game:
        return self.neural_net.game

    @property
    def num_mcts_sims(self):
        return self.default_mcts_params.num_mcts_sims

    @num_mcts_sims.setter
    def num_mcts_sims(self, value):
        self.default_mcts_params.num_mcts_sims = value

    def solve(self, state: State[TMove], num_mcts_sims: int | AZMctsParameters | None = None) -> TMove:
        self.neural_net.set_to_eval()
        solver = self.mcts_solver(num_mcts_sims)
        return solver.solve(state)

    def policy(
        self,
        state: State[TMove],
        num_mcts_sims: int | AZMctsParameters | None = None,
        is_raw_policy: bool = False,
    ) -> TMove:
        self.neural_net.set_to_eval()
        solver = self.mcts_solver(num_mcts_sims)
        return solver.policy(state, is_raw_policy)

    def self_play(
        self, training_params: AZTrainingParameters, initial_state: str | list[int] | None = None
    ) -> list[TrainingData]:
        state: State = self.game.initial_state(initial_state)
        status = state.status()

        solver = self.mcts_solver(training_params.mcts_parameters)

        self.neural_net.set_to_eval()
        policy_history: list[RecordedPolicy] = []

        while status.is_in_progress:
            state_before = state
            policy, _ = solver.policy(state_before)
            move = state_before.select_move(policy, training_params.temperature)
            state = copy(state_before)
            state.play_move(move)
            recorded_policy = RecordedPolicy(state_before, policy, move, state)
            policy_history.append(recorded_policy)
            status = state.status()

        training_set: list[TrainingData] = []
        for ph in policy_history:
            encoded_state = ph.state_before.to_numpy()
            perspective = ph.state_after.played_by
            outcome = status.outcome(perspective)
            training_data = TrainingData(encoded_state, ph.policy, outcome)
            training_set.append(training_data)

        return training_set

    def self_play_parallel(
        self, training_params: AZTrainingParameters, initial_state: str | list[int] | None = None
    ) -> list[TrainingData]:
        init_state: State = self.game.initial_state(initial_state)
        initial_status = init_state.status()
        in_progress = initial_status.is_in_progress
        if not in_progress:
            return []

        solver = self.mcts_solver(training_params.mcts_parameters)

        num_parallel = training_params.num_games_in_parallel
        parallel_games = [ParallelGame(i, init_state) for i in range(num_parallel)]
        in_progress_games = [pg for pg in parallel_games]

        self.neural_net.set_to_eval()

        while in_progress_games:
            states = [pg.latest_state for pg in in_progress_games]
            policies = solver.policies(states)

            for i, pg in enumerate(in_progress_games):
                state_before = pg.latest_state
                policy = policies[i]
                move = state_before.select_move(policy, training_params.temperature)
                state = copy(state_before)
                state.play_move(move)
                recorded_policy = RecordedPolicy(state_before, policy, move, state)
                pg.policy_history.append(recorded_policy)

            in_progress_games = [g for g in in_progress_games if g.latest_state.status().is_in_progress]

        training_set: list[TrainingData] = []
        for pg in parallel_games:
            terminal_status = pg.latest_state.status()
            policy_history = pg.policy_history
            for ph in policy_history:
                encoded_state = ph.state_before.to_numpy()
                perspective = ph.state_after.played_by
                outcome = terminal_status.outcome(perspective)
                training_data = TrainingData(encoded_state, ph.policy, outcome)
                training_set.append(training_data)

        return training_set

    def self_learn(
        self,
        train_params: AZTrainingParameters,
        initial_state: str | list[int] | None = None,
    ) -> None:
        for _ in trange(train_params.num_generations, desc="Generations"):
            training_set: list[TrainingData] = []

            num_rounds = train_params.games_per_generation // train_params.num_games_in_parallel
            for _ in trange(num_rounds, desc="- Self-play", leave=False):
                training_set += self.self_play(train_params, initial_state)
                # training_set += self.self_play_parallel(train_params, initial_state)

            extended_training_set = self._exploit_symmetries(training_set)

            for _ in trange(train_params.num_epochs, desc=" - Training", leave=False):
                self.neural_net.train(extended_training_set, train_params.minibatch_size)

            self.neural_net.generation += 1
            self.neural_net.save()

    def _exploit_symmetries(self, training_set: list[TrainingData]) -> list[TrainingData]:
        def all_sets(data: TrainingData) -> list[TrainingData]:
            symmetries = self.game.symmetries(data.encoded_state, data.policy)
            return (TrainingData(state, pol, data.outcome) for state, pol in symmetries)

        return [sym_data for training_data in training_set for sym_data in all_sets(training_data)]

    def mcts_solver(self, num_mcts_sims: int | AZMctsParameters | None = None) -> AlphaZeroMctsSolver:
        if num_mcts_sims is None:
            value_by_prop = asdict(self.default_mcts_params)
        elif isinstance(num_mcts_sims, int):
            value_by_prop = asdict(self.default_mcts_params)
            value_by_prop["num_mcts_sims"] = num_mcts_sims
        else:
            value_by_prop = asdict(num_mcts_sims)

        return AlphaZeroMctsSolver(self.neural_net, **value_by_prop)
