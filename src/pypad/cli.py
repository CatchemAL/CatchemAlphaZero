from .mnist_loader import load_data_wrapper
from .mnist_svm import svm_baseline

def main() -> None:

    svm_baseline()

    training_data, validation_data, test_data = load_data_wrapper()
    print("Hello world")