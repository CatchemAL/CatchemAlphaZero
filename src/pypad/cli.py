from .mnist_loader import load_data_wrapper
from .mnist_mlp import mlp_run
from .mnist_svm import svm_baseline
from .mnist_torch import run_torch

def main() -> None:

    run_torch()
    return
    mlp_run()
    svm_baseline()

    print("Hello world")