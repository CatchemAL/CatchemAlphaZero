<img src="https://raw.githubusercontent.com/CatchemAl/CatchemAlphaZero/main/src/caz/images/caz_splashscreen.jpg" width="420">

## Features

[![example workflow](https://github.com/CatchemAl/Doddle/actions/workflows/python-app.yml/badge.svg)](https://github.com/CatchemAl/Doddle/actions)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/catchem-alpha-zero)](https://pypi.org/project/catchem-alpha-zero/#files)
[![PyPI](https://img.shields.io/pypi/v/catchem-alpha-zero.svg)](https://pypi.org/project/catchem-alpha-zero/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/catchem-alpha-zero)
[![Tutorial](https://img.shields.io/badge/CAZ-tutorial-orange?logo=jupyter)](https://github.com/CatchemAL/CatchemAlphaZero/tree/main/tutorials)

## Installation
`pip install catchem-alpha-zero`

### Project Overview
`CatchemAlphaZero` (CAZ) showcases techniques in artificial intelligence for solving two-player games without any human knowledge or input except for the rules of the game.  Any two-player deterministic game can, in principle, leverage this framework.  Little is needed beyond the ability to: 
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
 - **Watch:** Click [here](https://youtu.be/HdjE3QIr12A) to watch a demo of the application.



### How strong is CAZ?
The current [Elo](https://en.wikipedia.org/wiki/Elo_rating_system) of CAZ is not precisely known (and obviously varies with increasing number of MCTS simulations). The ultimate plan is to integrate CAZ as a Lichess bot. For now, the strength of CAZ has been calculated by playing friends and colleagues who (are far better at chess than me and) have generously offered to play the bot. To date, CAZ has not lost on 1,000 simulations and [Chess.com](https://www.chess.com) consistently estimates its Elo to be in the range 1,900 to 2,300. My suspicion is that it is at the lower end of this spectrum - like AlphaZero, CAZ is strong at strategic play and closed positions but struggles with tactical play and endgames (where technique matters more).

At the time of writing, the 'state of the art' is to use an Efficiently Updating Sparse Flat Neighborhood Network (SF NNUE). This is a smaller, faster CPU bound network that allows for deeper searches and therefore shines in tactical play as well as strategic play. This project uses a large convolutional ResNet architecture exactly as per DeepMind's specification. It works best on a GPU (but CPU only runs are supported as well).

<div style="text-align:center">
  <img src="https://raw.githubusercontent.com/CatchemAl/LargeFiles/main/CAZ/CAZvAlvar.gif" width="420">
  <figcaption><br/>A friend (white) vs CAZ (Black) 0-1 under classical conditions. CAZ wins via resignation. Chess.com <a href="https://www.chess.com/analysis/library/jAYnm3Tdx">estimated</a> the Elo scores as 1,800 (white), 2,350 (black).<br/></figcaption>
</div>
<br/>

It's possible to train CAZ directly but if you'd like to use the weights that I have produced, you can download them from [here](https://github.com/CatchemAL/LargeFiles/tree/main/CAZ). CAZ assumes that the weights are saved in a `weights` folder within the active current directory.


### How it works
CatchemAlphaZero is a project that explores artificial intelligence techniques for two player games. The project started with [minimax](https://en.wikipedia.org/wiki/Minimax), which was then extended to [alpha-beta](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) minimax. [MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) was added to support games where leaf nodes could not be reached via brute force search. AlphaZero extends MCTS by using neural networks to both guide the search and provide and evaluation of each position rather than entering the rollout phase.

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

CAZ is able to learn games entirely through self-play reinforcement learning. The idea is to use MCTS to explore a sample of the search space that seems most promising. At each step, the MCTS is guided by a policy retrieved from the neural network and evaluates a position using the neural network's evalution result. The neural network is a two-headed (policy head and value head) [ResNet](https://en.wikipedia.org/wiki/Residual_neural_network) architecture that is described in detail in DeepMind's [Nature paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ). Below, is an illustration of the architecture used for TicTacToe. For chess, the architecture uses 19 residual network blocks:


<div style="text-align:center">
  <img src="https://raw.githubusercontent.com/CatchemAl/LargeFiles/main/CAZ/Tensorboard-ResNet.png" width="420">
  <figcaption><br/>Visualisation of the network architecture for TicTacToe using Tensorboard.</figcaption>
</div><br/>

To improve the neural network, CAZ learns through self-play. The steps are:
1. Play a game against itself relying on MCTS and the neural network
2. At each step, improve the current policy via MCTS - MCTS is a policy improvement operator
3. Record the outcome of the match after the game is played to completion
4. Play 100s of matches and batch the results together
5. Train the neural network against the match results. In particular, learn an improved policy based on the MCTS. Additionally, improve the network's evaluation function by assigning a score {-1, 0, +1} to **each state** depending upon the final outcome.


If you'd like to train the network yourself, check out the command line interface below.

### Command line Interface Features
CAZ exposes five entry points via the command line: `run`, `learn`, `kaggle`, `super`, `hyper`
1) **Run** CAZ from the terminal to see it play a game
2) **Learn** through self-play if you wish to train a netowrk
3) **Kaggle** enables you to run a Kaggle compliant agent that complies with the Kaggle [Simulation Competitions](https://www.kaggle.com/competitions/connectx) API
4) **super** trains a neural network in chess through supervised learning. Whilst it is entirely possible to learn solely through self-play reinforcement learning, it would take a very long time to do this exclusively for chess (Deepmind used 5,000 TPUs for training which is out of my budget!). Supervised learning was used to accelerate the learning process specifically for chess but not for any other game.
5) **Hyper** performs a search over CAZ's various hyperparameters to ensure that it is learning optimally as training progresses 

Examples of how to run the various entry points are shown below:
- `caz run --game=tic --player1=mcts --player2=az # MCTS vs AlphaZero algorithm`.
- `caz run --game=connectx --player1=mcts --player2=mcts`
- `caz run --game=chess --player1=az --player2=az`
- `caz learn --game=connectx`
- `caz kaggle --game=connectx --player1=mcts --player2=mcts`
- `caz super`
- `caz hyper --game=connectx --gen=53`


## Chess Representation
CAZ follows Deepmind's proposal for representing input features and the action space per the [follow up paper](https://arxiv.org/pdf/1712.01815.pdf). In particular, a state is represented by an 8x8x20 input tensor where the 20 planes correspond to:
1. Current player's pawn positions 
1. Current player's rook positions 
1. Current player's knight positions 
1. Current player's bishop positions 
1. Current player's queen positions 
1. Opposing player's king positions
1. Opposing player's pawn positions 
1. Opposing player's rook positions 
1. Opposing player's knight positions 
1. Opposing player's bishop positions 
1. Opposing player's queen positions 
1. Opposing player's king positions
1. 8x8 matrix of 0's or 1's denoting the player's turn (all zeroes or all ones)
1. 8x8 matrix of 0's or 1's denoting current player's kingside castling rights
1. 8x8 matrix of 0's or 1's denoting current player's queenside castling rights
1. 8x8 matrix of 0's or 1's denoting opposing player's kingside castling rights
1. 8x8 matrix of 0's or 1's denoting opposing player's queenside castling rights
1. 8x8 matrix of showing the halfmove clock
1. indicators for the position of any en passant squares
1. 8x8 matrix denoting whether the state is a two-fold repetition

The action space is an 8x8x73 tensor where each 8x8 plane corresponds to:
- x7 N moves (the maximum number of squares you can move north is 7)
- x7 NE moves (again, moving 1-7 moves in a north-east direction)
- x7 E moves
- x7 SE moves
- x7 S moves
- x7 SW moves
- x7 W moves
- x7 NW moves
- Knight move NNE (i.e. one square east and two north)
- Knight move ENE (i.e. two square east and two north)
- Knight move ESE
- Knight move SSE
- Knight move SSW
- Knight move WSW
- Knight move WNW
- Knight move NNW
- Promote NW Knight (underpromote to a knight by capturing north-west)
- Promote N Knight (underpromote to a knight by advancing north)
- Promote NE Knight
- Promote NW Rook
- Promote N  Rook
- Promote NE Rook
- Promote NW Bishop
- Promote N  Bishop
- Promote NE Bishop


<p float="center">
  <img src="https://raw.githubusercontent.com/CatchemAl/LargeFiles/main/CAZ/CAZ_Chess_Space.png" width="420">
  <figcaption>Visualisation of the observation space (left) and action space (right). Here, CAZ is set to show the top two moves which align with common theory for the Fried Liver Attack. You can tweak the slider to show more/fewer moves or interrogate top moves by action plane.</figcaption>
</p>


### Tutorial
CAZ is not, nor ever intends to be a top chess engine\*. Rather, it is a showcase in AI techniques and my attempt to reproduce Deepmind's work. If you'd like to know more about the inner working of self-play reinforcement learning, check out the tutorials. They contain beautifully rendered visualisations to help you understand the way the project works. The tutorials have been organised into four sections:
1. **MCTS Demo** which showcases MCTS including the UCB formula. You can render search trees to see it in action
2. **Convolutional Layers** which shows how convolutional layers work and can combine to transform the 'images' of the game
3. **Alpha Zero** where you can see CatchemAlphaZero's policies and play against it within an IPython setting
4. Chess - which illustrates the exact details of the chess setup. You can interrogate the network and play againt it here (although GUI app preferred)

\* However, I am a glutton for punishment and will probably rewrite this project in C++ to make it stronger at some point...


Thanks for making it this far!

CatchemAL


