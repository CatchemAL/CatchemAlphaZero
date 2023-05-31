from .mnist_mlp import mlp_run
from .mnist_svm import svm_baseline
from .mnist_torch import run_torch


def main() -> None:
    run_torch()
    mlp_run()
    svm_baseline()


if __name__ == "__main__":
    main()
