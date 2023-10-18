import sys
import os

from securityGPT.dataset import Loader
from typing import List, Tuple, Generator
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

import numpy as np

dataset_path = os.path.join("../../data")
dataset_folder = os.path.join("../../models/svm")

class SVM:
    """
    A wrapper class for Support Vector Machine (SVM) provided by scikit-learn.

    Parameters:
    - kernel (str): The SVM kernel type ('linear', 'poly', 'rbf', 'sigmoid', etc.). Default is 'linear'.
    - C (float): The regularization parameter that controls the trade-off between margin and classification error. Default is 1.0.

    Attributes:
    - svm (svm.SVC): The scikit-learn SVM classifier.

    Methods:
    - train(X: np.ndarray, y: np.ndarray) -> None:
        Train the SVM model on the provided training data.

    - predict(X: np.ndarray) -> np.ndarray:
        Make predictions on the input data using the trained SVM model.

    - stats(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        Calculate accuracy and F1 score based on true labels and predicted labels.
    """

    def __init__(self, kernel : str = 'linear', C : float = 1.0, SGD : bool = False) -> None:
        if SGD:
            self.svm = SGDClassifier(loss='hinge', alpha=0.01, max_iter=100)
        else:
            self.svm =  svm.SVC(kernel=kernel, C=C)

    def train(self, X : np.ndarray, y : np.ndarray):
        """
        Train the SVM model on the provided training data.

        Parameters:
        - X (np.ndarray): The training data features as a NumPy array.
        - y (np.ndarray): The training data labels as a NumPy array.

        Returns:
        - None
        """
        for X_train, y_train in zip(X, y): 
            self.svm.fit(X_train, y_train)
    
    def train(self, data : Generator[Tuple[np.ndarray, np.ndarray], None, None]) -> None:
        """
        Train the SVM model on the provided training data.

        Parameters:
        - data (np.ndarray) : The training data generator that yields X_train, and y_train.

        Returns:
        - None
        """
        for X_train, y_train in data(): 
            try: 
                self.svm.fit(X_train, y_train)
            except:
                continue

    def predict(self, x : np.ndarray) -> np.ndarray:
        """
        Make predictions on the input data using the trained SVM model.

        Parameters:
        - X (np.ndarray): The input data features as a NumPy array.

        Returns:
        - np.ndarray: The predicted labels as a NumPy array.
        """
        return self.svm.predict(x)

    def stats(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """
        Calculate accuracy and F1 score based on true labels and predicted labels.

        Parameters:
        - y_true (np.ndarray): The true labels as a NumPy array.
        - y_pred (np.ndarray): The predicted labels as a NumPy array.

        Returns:
        - Tuple[float, float]: A tuple containing accuracy and F1 score.
        """
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return acc, f1

if __name__ == "__main__":
    size = 5000
    ratio = 0.1
    SGD = True #Use Gradient Descent (better for larger scale datasets)
    dataloader = Loader(dataset_path, size=size)
    X_train, y_train, X_test, y_test, data = dataloader.load(bootstrap=True, ratio=ratio)
    kernel = 'linear'
    svm = SVM(kernel=kernel, SGD=SGD)
    svm.train(data)
    y_pred = svm.predict(X_test)
    acc, f1 = svm.stats(y_test, y_pred)  # Corrected here
    print(f"Accuracy: [{acc}] \nF1 Score: [{f1}]")

