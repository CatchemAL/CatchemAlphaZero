<img src="https://raw.githubusercontent.com/CatchemAl/CatchemAlphaZero/main/src/caz/icons/caz_splashscreen.jpg" width="420">

## Features

[![example workflow](https://github.com/CatchemAl/Doddle/actions/workflows/python-app.yml/badge.svg)](https://github.com/CatchemAl/Doddle/actions)
[![codecov](https://codecov.io/gh/CatchemAl/Doddle/branch/main/graph/badge.svg?token=3JM8LJ3IKS)](https://codecov.io/gh/CatchemAl/Doddle)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/Doddle)](https://pypi.org/project/doddle/#files)
[![PyPI](https://img.shields.io/pypi/v/doddle.svg)](https://pypi.org/project/doddle/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Doddle)
[![Tutorial](https://img.shields.io/badge/doddle-tutorial-orange?logo=jupyter)](https://github.com/CatchemAl/Doddle/blob/main/tutorial/Getting%20Started.ipynb)

## Installation
`pip install catchem-alpha-zero`

### Project Overview
`CatchemAlphaZero` (CAZ) showcases techniques in artificial intelligence for solving two-player games without any human knowledge or input except for the rules of the game.  Any two-player deterministic game can,  in principle, leverage this framework.  Little is needed beyond the ability to: 
 1. get the games current legal moves
 2. play a move
 3. determine if the game has terminated.

The project includes:
 - [Alpha-Beta Minimax](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) which is well suited to games with a small to medium sized decision tree (or games for which one can easily evaluate a position's likelihood of winning)
 - [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)  which uses random simulation to estimate the value of a state,  and biases the search towards paths that seem most promising
 - [AlphaZero](https://en.wikipedia.org/wiki/AlphaZero): an attempt to replicate the brilliant work of [DeepMind](https://www.deepmind.com/), who used deep convolutional neural networks in conjunction with Monte Carlo Tree SearchTo significantly improve upon the MCTS algorithm.  The neural network learns the optimal policy entirely through self-play reinforcement learning.

### Supported Games

- Chess
- ConnectX ([Connect 4](https://en.wikipedia.org/wiki/Connect_Four) for boards of arbitrary sizes)
- Tic Tac Toe

  

### Play Chess Against `CatchemAlphaZero`

The crowning glory of this project is chess. `CatchemAlphaZero` bundles a GUI application with the wheel, `alpha`, installed into your virtual environments Scripts folder. To launch the GUI, either click the application within the scripts folder or type `alpha` into the command line (having activated your virtual environment). The application will launch with a splashscreen presenting the following options:
- Play as White
- Play as Black
- Play as Both
- Spectate Mode (watch CAZ play against itself) 

  <img src="https://raw.githubusercontent.com/CatchemAl/LargeFiles/main/CAZ/GUI_demo.png" width="420">
  
#### Key features
 - **MCTS Num Sims:** Increase the size of the Monte Carlo Tree Search to increase the difficulty (1,000 sims corresponds to a chess ELO of about 1,800).
 - **Show Eval Bar** to see how CAZ evaluates the current position
 - **Asynchronous Ponder:** CAZ is fully asynchronous and will ponder on your thinking time. CAZ efficiently recycles the tree search between moves and explores whilst its opponent is thinking. Take time to think but remember that CAZ is thinking too!
 - **Watch:** Click here to watch a demo of the application.



### How strong is CAZ?
The current [Elo](https://en.wikipedia.org/wiki/Elo_rating_system) of CAZ is not precisely known (and obviously varies with increasing number of MCTS simulations). The ultimate plan is to integrate CAZ as a Lichess bot. For now, the strength of CAZ has been calculated by playing friends and colleagues who have generously offered to play the bot. To date, CAZ has not lost on 1,000 simulations and Chess.com consistently estimates its Elo to be in the range 1,900 to 2,300. My suspicion is that it is at the lower end of this figure - like AlphaZero, CAZ is strong at strategic play and openings but struggles with tactical play and endgames (where technique matters more).

At the time of writing, the 'state of the art' is to use an Efficiently Updating Sparse Flat Neighborhood Network (SF NNUE). This is a smaller, faster CPU bound network that allows for deeper searches and therefore shines in tactical play as well as strategic play. This project uses a large convolutional ResNet architecture exactly as per DeepMind's specification. It works best on a GPU (but CPU only runs are supported as well).

<div style="text-align:center">
  <img src="https://raw.githubusercontent.com/CatchemAl/LargeFiles/main/CAZ/CAZvAlvar.gif" width="420">
  <figcaption><br/>A friend (white) vs CAZ (Black) 0-1 under classical conditions. CAZ wins via resignation. Chess.com <a href="https://www.chess.com/analysis/library/jAYnm3Tdx">estimated</a> the Elo scores as 1,800 (white), 2,350 (black).<br/></figcaption>
</div>
<br/>

It's possible to train CAZ directly but if you'd like to use the weights that I have produced, you can download them from here. CAZ assumes that the weights are saved in a `weights` folder within the active current directory.


### How it works
CatchemAlphaZero is project that explores artificial intelligence techniques for two player games. The project started with minimax, which was then extended to alpha-beta minimax. MCTS was added to support games where leaf nodes could not be reached via brute force search. AlphaZero extends MCTS by using neural networks to both guide the search and provide and evaluation of each position rather than entering the rollout phase.

CAZ produces beautiful visualisations for each game. In particular, it is possible to render the full tree search as well as the policy & evaluation associated with each state. The image below shows the output of a tree search with 10 simulations (please note that CAZ assumes that you have the [GraphViz](https://graphviz.org/doc/info/command.html) application already installed.)

<div style="text-align:center">
  <img src="https://raw.githubusercontent.com/CatchemAl/LargeFiles/main/CAZ/graph.svg" width="420">
  <figcaption><br/>Rendered with ❤️ using CAZ. See the tutorial for details on how.</figcaption>
</div><br/>


CAZ also renders each game state as HTML and will show the policy associated with each state as an HTML grid with a heatmap overlay. The darker squares correspond to squares favoured by the policy. The evaluation shows that CAZ believes it has a 100% chance of winning from here and correctly identifies the three moves that can lead to a win.


<div style="text-align:center">
  <img src="https://raw.githubusercontent.com/CatchemAl/LargeFiles/main/CAZ/TTT%20Policy.png" width="420">
  <figcaption><br/>Heatmap policy for Tic-Tac-Toe. States and policied states are understood natively by IPython.</figcaption>
</div><br/>

# Learning through self-play reinforcement learning

CAZ is able to learn games entirely through self-play reinforcement learning. The premise 


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

