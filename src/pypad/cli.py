from .mnist_loader import load_data_wrapper
from .mnist_mlp import mlp_run
from .mnist_svm import svm_baseline

def main() -> None:

    mlp_run()
    svm_baseline()

    print("Hello world")