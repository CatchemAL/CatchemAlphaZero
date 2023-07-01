import pickle
from pathlib import Path

from ..games import ConnectX, Game
from .alpha_zero_parameters import AZTrainingParameters
from .network import TrainingData
from .network_torch import PytorchNeuralNetwork


def learn_from_data(
    generation: int, training_data: list[TrainingData], training_params: AZTrainingParameters
):
    neural_net.train(training_data, training_params.num_epochs, training_params.minibatch_size)
    neural_net.generation = generation + 1
    neural_net.save()


def get_training_files(game: Game, directory: Path | str) -> None:
    directory = Path(directory)

    training_regex = f"{game.fullname}_gen*.pkl"
    training_files = [file_path for file_path in directory.glob(training_regex)]

    for training_file in training_files:
        generation = int(training_file.stem.split("gen")[1])
        with open(training_file, "rb") as f:
            data = pickle.load(f)

        yield (generation, data)


if __name__ == "__main__":
    game = ConnectX()
    training_params = AZTrainingParameters.defaults(game.fullname)
    neural_net = PytorchNeuralNetwork.create(game, "..")

    for gen, data in get_training_files(game, r"../training_data"):
        print(gen)
        learn_from_data(gen, data, training_params)
