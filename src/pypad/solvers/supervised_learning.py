import math

import numpy as np
from chess import Move
from chess.engine import Limit, Score, SimpleEngine
from tqdm import trange

from ..games.chess import Chess, ChessState
from ..solvers.alpha_zero_parameters import AZTrainingParameters
from .network import NeuralNetwork, TrainingData


class SupervisedTrainer:
    def __init__(self, neural_net: NeuralNetwork, engine: SimpleEngine) -> None:
        self.neural_net = neural_net
        self.engine = engine

    def train(self, training_params: AZTrainingParameters) -> None:
        num_epochs, minibatch_size = training_params.num_epochs, training_params.minibatch_size

        for i in trange(training_params.num_generations):
            training_data = self.generate_training_data(10_000)

            extended_training_set = self._exploit_symmetries(training_data)

            # Train against the newly generated games
            if i <= 10 or i % 10 == 0:
                self.neural_net.save_training_data(extended_training_set)
            self.neural_net.train(extended_training_set, num_epochs, minibatch_size)
            self.neural_net.generation += 1
            self.neural_net.save()

    def _exploit_symmetries(self, training_data) -> list[TrainingData]:
        # todo
        return training_data

    def generate_training_data(self, min_num_points: int):
        training_set: list[TrainingData] = []
        while len(training_set) < min_num_points:
            state: ChessState = self.neural_net.game.initial_state()
            while state.status().is_in_progress:
                training_data, top_moves = self.to_data_point(state)
                training_set.append(training_data)

                idx = np.random.choice(len(top_moves))
                top_move = top_moves[idx]
                state.set_move(top_move)

        return training_set

    def build_policy(self, state: ChessState, top_moves: list[Move], noise: float) -> np.ndarray:
        action_size = self.neural_net.game.config().action_size
        encoded_policy = np.zeros(action_size, dtype=np.float32)

        prior = 1
        decay = 0.8
        for move in top_moves:
            loc = state.policy_loc(move)
            encoded_policy[loc] = prior
            prior *= decay

        for move in state.status().legal_moves:
            loc = state.policy_loc(move)
            encoded_policy[loc] += noise

        encoded_policy /= encoded_policy.sum()
        return encoded_policy

    def to_data_point(self, state: ChessState) -> tuple[TrainingData, list[Move]]:
        time_limit = Limit(time=0.01)

        infos = self.engine.analyse(state.board, time_limit, multipv=5)
        top_moves = [info["pv"][0] for info in infos]
        encoded_state = state.to_feature()
        score = infos[0]["score"]

        if score.relative.is_mate():
            outcome = 1.0
        else:
            cp = score.relative.score()
            outcome = SupervisedTrainer.centipawn_to_q_value(cp)

        policy = self.build_policy(state, top_moves, 0.5)
        training_data = TrainingData(encoded_state, policy, outcome)
        return training_data, top_moves

    @staticmethod
    def centipawn_to_q_value(cp: int) -> float:
        return math.atan(cp / 111.714640912) / 1.5620688421

    @staticmethod
    def q_value_to_centipawn(q: float) -> int:
        cp = 111.714640912 * math.tan(1.5620688421 * q)
        return round(cp)
