import math
import random

import numpy as np
from chess import Move
from chess.engine import Limit, PovScore, SimpleEngine
from tqdm import trange

from ..solvers.alpha_zero_parameters import AZTrainingParameters
from ..states.chess import ChessState
from .network import NeuralNetwork, TrainingData


class SupervisedTrainer:
    def __init__(self, neural_net: NeuralNetwork, engine: SimpleEngine) -> None:
        self.neural_net = neural_net
        self.engine = engine

    def train(self, training_params: AZTrainingParameters) -> None:
        discount_factor = training_params.mcts_parameters.discount_factor
        num_epochs, minibatch_size = training_params.num_epochs, training_params.minibatch_size

        for i in trange(training_params.num_generations):
            training_data = self.generate_training_data(10_000, discount_factor)

            extended_training_set = self._exploit_symmetries(training_data)

            # Train against the newly generated games
            self.neural_net.train(extended_training_set, num_epochs, minibatch_size)
            self.neural_net.generation += 1

            if i % 10 == 0:
                self.neural_net.save_training_data(extended_training_set)
                self.neural_net.save()

    def _exploit_symmetries(self, training_data) -> list[TrainingData]:
        # todo
        return training_data

    def generate_training_data(self, min_num_points: int, discount_factor: float):
        BLUNDER_PROBABILITY = 0.2

        training_set: list[TrainingData] = []
        while len(training_set) < min_num_points:
            state: ChessState = self.neural_net.game.initial_state()
            status = state.status()
            while status.is_in_progress:
                training_data, top_moves = self.to_data_point(state, discount_factor)
                training_set.append(training_data)

                if random.random() < BLUNDER_PROBABILITY:
                    legal_moves = status.legal_moves
                    idx = np.random.choice(len(legal_moves))
                    move = legal_moves[idx]
                    state.set_move(move)
                else:
                    idx = np.random.choice(len(top_moves))
                    top_move = top_moves[idx]
                    state.set_move(top_move)

                status = state.status()

        return training_set

    def build_policy(
        self, state: ChessState, top_moves: list[Move], top_scores: np.ndarray, noise: float
    ) -> np.ndarray:
        action_size = self.neural_net.game.config().action_size
        encoded_policy = np.zeros(action_size, dtype=np.float32)

        priors = SupervisedTrainer.policy_weights(top_scores)
        for move, prior in zip(top_moves, priors):
            loc = state.policy_loc(move)
            encoded_policy[loc] = prior

        for move in state.status().legal_moves:
            loc = state.policy_loc(move)
            encoded_policy[loc] += noise

        encoded_policy /= encoded_policy.sum()
        return encoded_policy

    def to_data_point(
        self, state: ChessState, discount_factor: float
    ) -> tuple[TrainingData, list[Move]]:
        encoded_state = state.to_feature()

        time_limit = Limit(time=0.01)
        infos = self.engine.analyse(state.board, time_limit, multipv=5)

        top_moves = [info["pv"][0] for info in infos]
        top_scores = np.array([SupervisedTrainer.score_to_centipawns(info["score"]) for info in infos])
        score = top_scores.max()
        outcome = SupervisedTrainer.centipawns_to_q_value(score)

        NOISE = 0.001
        policy = self.build_policy(state, top_moves, top_scores, NOISE)
        training_data = TrainingData(encoded_state, policy, outcome)
        return training_data, top_moves

    @staticmethod
    def score_to_centipawns(score: PovScore):
        if score.is_mate():
            full_moves = 2 * np.abs(score.relative.moves) - (1 if score.relative.moves > 0 else 0)
            value = 0.99**full_moves
            signed_value = np.sign(score.relative.moves) * value
            return SupervisedTrainer.q_value_to_centipawns(signed_value)
        return score.relative.score()

    @staticmethod
    def policy_weights(scores: np.ndarray) -> np.ndarray:
        drops = (scores.max() - scores) / 100.0
        policies = 0.15**drops
        return policies

    @staticmethod
    def centipawns_to_q_value(cp: int) -> float:
        return math.atan(cp / 250) / 1.5620688421

    @staticmethod
    def q_value_to_centipawns(q: float) -> int:
        cp = 250 * math.tan(1.5620688421 * q)
        return round(cp)
