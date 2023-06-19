<img src="https://raw.githubusercontent.com/CatchemAl/CatchemAlphaZero/main/src/caz/icons/caz_splashscreen.jpg" width="420">

## Features

[![example workflow](https://github.com/CatchemAl/Doddle/actions/workflows/python-app.yml/badge.svg)](https://github.com/CatchemAl/Doddle/actions)
[![codecov](https://codecov.io/gh/CatchemAl/Doddle/branch/main/graph/badge.svg?token=3JM8LJ3IKS)](https://codecov.io/gh/CatchemAl/Doddle)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/Doddle)](https://pypi.org/project/doddle/#files)
[![PyPI](https://img.shields.io/pypi/v/doddle.svg)](https://pypi.org/project/doddle/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Doddle)
[![Tutorial](https://img.shields.io/badge/doddle-tutorial-orange?logo=jupyter)](https://github.com/CatchemAl/Doddle/blob/main/tutorial/Getting%20Started.ipynb)

### Project Overview
`CatchemAlphaZero` showcases techniques in artificial intelligence for solving two-player games without any human knowledge or input except for the rules of the game.  Any two-player deterministic game can,  in principle, leverage this framework.  Little is needed beyond the ability to: 
 1. get the games current legal moves
 2. play a move
 3. determine if the game has terminated.

The project includes:
 - [Alpha-Beta Minimax](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) which is well suited to games with a small to medium sized decision tree
 - [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)  which uses random simulation to estimate the value of a state,  and biases the search towards paths that seem most promising
 - [AlphaZero](https://en.wikipedia.org/wiki/AlphaZero): an attempt to replicate the brilliant work of [DeepMind](https://www.deepmind.com/), who used deep convolutional neural networks in conjunction with Monte Carlo Tree SearchTo significantly improve upon the MCTS algorithm.  The neural network learns the optimal policy entirely through self-play reinforcement learning.

### Supported Games
- Chess
- ConnectX ([Connect 4](https://en.wikipedia.org/wiki/Connect_Four) for boards of arbitrary sizes)
- Tic Tac Toe

### Play Chess Against `CatchemAlphaZero`
`CatchemAlphaZero`  bundles a GUI  application with the wheel, `alpha`, installed into your virtual environments Scripts folder.


### Command line Interface Features
Doddle exposes four entry points via the command line: `run`, `solve`, `hide`, `benchmark`
1)  **Run** the solver to see how the game is optimally played
2) **Solve** a game in realtime using Doddle's solver
3) Play a variation of the game where the solver attempts to **hide** the answer from you for as long as possible (inspired by [Absurdle](https://qntm.org/files/absurdle/absurdle.html))
4) **Benchmark** Doddle against the entire dictionary to see how well it performs

The commands can be run with additional parameters:
- Play using words of length 4-9 (inclusive) by adding the optional `--size` parameter (default is 5).
- Choose your solver using the `--solver=ENTROPY` or `--solver=MINIMAX` parameter (default is minimax)
- Run deep searches using the `--depth` parameter (default is 1)
- Solve multiple games of Wordle at the same time. This mode is inspired by popular spin-offs such as [Dordle](https://zaratustra.itch.io/dordle), [Quordle](https://www.quordle.com/#/) and [Octordle](https://octordle.com/). Playing multiple games with Doddle is easy: just add more answers to the run command `doddle run --answer=ULTRA,QUICK,SOLVE` and Doddle will solve them all at the same time.

