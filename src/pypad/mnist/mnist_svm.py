"""
A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""

import matplotlib.pyplot as plt
from sklearn import svm, metrics

from .mnist_loader import load_data


def svm_baseline():
    training_data, validation_data, test_data = load_data()

    x_train, y_train = training_data[0][:1_000, :], training_data[1][:1_000]
    x_test, y_test = test_data[0], test_data[1]

    # train
    print("Begin SVM training")
    clf = svm.SVC()
    clf.fit(x_train, y_train)

    # test
    predictions = clf.predict(x_test)
    is_correct = predictions == y_test
    num_correct = is_correct.sum()
    print("Baseline classifier using an SVM.")
    print(str(num_correct) + " of " + str(len(y_test)) + " values correct.")

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predictions)}\n"
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()
