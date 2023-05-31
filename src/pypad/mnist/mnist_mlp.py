"""
A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

from .mnist_loader import load_data


def mlp_run():
    training_data, validation_data, test_data = load_data()

    x_train, y_train = training_data[0][:, :], training_data[1][:]
    x_test, y_test = test_data[0], test_data[1]

    # train
    print("Begin MLP training")
    mlp = MLPClassifier(
        hidden_layer_sizes=(125,),
        alpha=0.0005,
        solver="sgd",
        verbose=10,
        random_state=1,
        learning_rate_init=0.22,
    )

    mlp.fit(x_train, y_train)

    # test
    predictions = mlp.predict(x_test)
    is_correct = predictions == y_test
    num_correct = is_correct.sum()
    print("Baseline classifier using an MLP.")
    print(str(num_correct) + " of " + str(len(y_test)) + " values correct.\n")

    return

    print(
        f"Classification report for classifier {mlp}:\n"
        f"{metrics.classification_report(y_test, predictions)}\n"
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()
