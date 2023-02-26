"""
mnist_svm
~~~~~~~~~
A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""

import matplotlib.pyplot as plt
from sklearn import svm, metrics

from .mnist_loader import load_data


def svm_baseline():
    training_data, validation_data, test_data = load_data()
    
    # train
    print('Begin SVM training')
    clf = svm.SVC()
    clf.fit(training_data[0][:10_000,:], training_data[1][:10_000])
    
    # test
    x_test, y_test = test_data[0], test_data[1]
    predictions = clf.predict(test_data[0])
    is_correct = predictions == test_data[1]
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

if __name__ == "__main__":
    svm_baseline()