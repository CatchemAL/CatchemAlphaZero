import sys
from argparse import ArgumentParser, Namespace
from typing import Sequence

import numpy as np

from .games import GameType
from .solvers import AgentType, Solver

# from .factory import create_agent, get_controller, load_player

# from .mnist.mnist_mlp import mlp_run
# from .mnist.mnist_svm import svm_baseline
# from .mnist.mnist_torch import run_torch


def run(args: Namespace) -> None:
    game_type: GameType = args.game
    init: str = args.init
    player1_type: AgentType = args.player1
    player2_type: AgentType = args.player2

    game = game_type.create()
    state = game.initial_state(init)

    player1 = player1_type.create_player()
    player2 = player2_type.create_player()

    player, opponent = player1, player2
    while list(state.legal_moves()):
        move = player.solve(state)
        state.play_move(move)
        game.display(state)
        player, opponent = opponent, player

    game.display_outcome(state)


def kaggle(args: Namespace) -> None:
    from kaggle_environments import make

    game_type: GameType = args.game
    player1_type: AgentType = args.player1
    player2_type: AgentType = args.player2

    player1 = player1_type.create_player()
    player2 = player2_type.create_player()

    game = game_type.create()
    agent1 = game.create_agent(player1)
    agent2 = game.create_agent(player2)

    # Setup a ConnectX environment.
    env = make(game.label, debug=True)
    env.run([agent1, agent2])
    env.render(mode="ipython")


def main() -> None:
    mcts_az()
    tictactoe()
    mcts()
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

    run_torch()
    return
    mlp_run()
    svm_baseline()

    print("Hello world")


def main() -> None:
    parse_args(sys.argv[1:])


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

    namespace = parser.parse_args(args)
    namespace.func(namespace)
