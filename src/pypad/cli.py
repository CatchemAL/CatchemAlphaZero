import sys
from argparse import ArgumentParser, Namespace
from typing import Sequence

import numpy as np

from .games import GameType, get_game
from .solvers import AgentType, Solver


def run(args: Namespace) -> None:
    game_type: GameType = args.game
    init: str = args.init
    player1_type: AgentType = args.player1
    player2_type: AgentType = args.player2

    game = get_game(game_type)
    state = game.initial_state(init)

    player1 = player1_type.create_player(game)
    player2 = player2_type.create_player(game)

    player, opponent = player1, player2
    while state.status().is_in_progress:
        move = player.solve(state)
        state.set_move(move)
        game.display(state)
        player, opponent = opponent, player

    game.display_outcome(state)


def kaggle(args: Namespace) -> None:
    from kaggle_environments import make

    game_type: GameType = args.game
    player1_type: AgentType = args.player1
    player2_type: AgentType = args.player2

    game = get_game(game_type)

    player1 = player1_type.create_player(game)
    player2 = player2_type.create_player(game)
    agent1 = game.create_agent(player1)
    agent2 = game.create_agent(player2)

    # Setup a ConnectX environment.
    env = make(game.name, debug=True)
    env.run([agent1, agent2])
    env.render(mode="ipython")


def learn(args: Namespace) -> None:
    from .alpha_zero import AlphaZero, AZTrainingParameters, PytorchNeuralNetwork

    game_type: GameType = args.game
    init: str = args.init

    game = get_game(game_type)

    # Build alpha zero with latest weights
    neural_net = PytorchNeuralNetwork.create(game, ".")
    alpha_zero = AlphaZero(neural_net)

    training_params = AZTrainingParameters.defaults(game.fullname)
    alpha_zero.self_learn(training_params, init)

    print("done!")


def supervised_learning(args: Namespace) -> None:
    import chess

    from .alpha_zero import AZTrainingParameters, PytorchNeuralNetwork, SupervisedTrainer

    init: str = args.init

    game = get_game(GameType.CHESS)
    training_params = AZTrainingParameters.defaults(game.fullname)

    # Build alpha zero with latest weights
    neural_net = PytorchNeuralNetwork.create(game, ".")

    path = r".\stockfish\stockfish-windows-2022-x86-64-avx2.exe"
    with chess.engine.SimpleEngine.popen_uci(path) as stockfish:
        trainer = SupervisedTrainer(neural_net, stockfish)
        trainer.train(training_params)

    print("done!")


def hyper(args: Namespace) -> None:
    from .alpha_zero import AlphaZero, AZArenaParameters, PytorchNeuralNetwork

    generation: int = args.gen
    game_type: GameType = args.game
    game = get_game(game_type)

    # Build alpha zero with latest weights
    neural_net = PytorchNeuralNetwork.create(game, ".", generation)
    alpha_zero = AlphaZero(neural_net)
    arena_params = AZArenaParameters()

    alpha_zero.explore_hyperparameters(generation, arena_params)

    print("done!")


def main() -> None:
    parse_args(sys.argv[1:])

    return
    run_kaggle()

    ROWS, COLS = 6, 7
    moves = [1, 1, 2, 2, 3, 3]
    # 12211221122137477577675665566556
    sequence = "1,2,2,1,1,2,2,1,1,2,2,1,3,7,4,7,7,5,7,7,6,7,5,6,6,5,5,6,6,5,5,6".split(",")
    moves = [int(s) for s in sequence]
    board = ConnectX.create(ROWS, COLS, moves)
    solver = Solver()
    solver.minimax(board, -np.inf, np.inf)


def parse_args(args: Sequence[str]) -> None:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    # Runs an adversarial game
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--game", type=GameType.from_str, default=GameType.TICTACTOE)
    run_parser.add_argument("--player1", type=AgentType.from_str, default=AgentType.HUMAN)
    run_parser.add_argument("--player2", type=AgentType.from_str, default=AgentType.HUMAN)
    run_parser.add_argument("--init", type=str, default="")
    run_parser.set_defaults(func=run)

    # Runs an adversarial game on the kaggle platform
    run_parser = subparsers.add_parser("kaggle")
    run_parser.add_argument("--game", type=GameType.from_str, default=GameType.TICTACTOE)
    run_parser.add_argument("--player1", type=AgentType.from_str, default=AgentType.HUMAN)
    run_parser.add_argument("--player2", type=AgentType.from_str, default=AgentType.HUMAN)
    run_parser.set_defaults(func=kaggle)

    # Trains a deep neural net via reinforcement learning
    run_parser = subparsers.add_parser("learn")
    run_parser.add_argument("--game", type=GameType.from_str, default=GameType.TICTACTOE)
    run_parser.add_argument("--init", type=str, default=None)
    run_parser.set_defaults(func=learn)

    # Trains a deep neural net via supervised learning
    run_parser = subparsers.add_parser("super")
    run_parser.add_argument("--init", type=str, default=None)
    run_parser.set_defaults(func=supervised_learning)

    # Explores learning rate as a function of training hyperparameters
    run_parser = subparsers.add_parser("hyper")
    run_parser.add_argument("--game", type=GameType.from_str, default=GameType.TICTACTOE)
    run_parser.add_argument("--gen", type=int, default=0)
    run_parser.set_defaults(func=hyper)

    namespace = parser.parse_args(args)
    namespace.func(namespace)
