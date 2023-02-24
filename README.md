<img src="https://raw.githubusercontent.com/CatchemAl/Doddle/main/images/DoddleLogo.png" width="420">

## Features

[![example workflow](https://github.com/CatchemAl/Doddle/actions/workflows/python-app.yml/badge.svg)](https://github.com/CatchemAl/Doddle/actions)
[![codecov](https://codecov.io/gh/CatchemAl/Doddle/branch/main/graph/badge.svg?token=3JM8LJ3IKS)](https://codecov.io/gh/CatchemAl/Doddle)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/Doddle)](https://pypi.org/project/doddle/#files)
[![PyPI](https://img.shields.io/pypi/v/doddle.svg)](https://pypi.org/project/doddle/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Doddle)
[![Tutorial](https://img.shields.io/badge/doddle-tutorial-orange?logo=jupyter)](https://github.com/CatchemAl/Doddle/blob/main/tutorial/Getting%20Started.ipynb)

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

