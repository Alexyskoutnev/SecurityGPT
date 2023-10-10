import sys
import os

from securityGPT.dataset import Loader
from typing import List
from sklearn import svm

import numpy as np


dataset_path = os.path.join("../../data")

class SVM:
    """Wrapper class for SVM provided by sklearn
    """

    def __init__(self, kernel : str = 'linear', C : float = 1.0) -> None:
        self.svm =  svm.SVC(kernel=kernel, C=C)

    def train(self, X : np.ndarray, y : np.ndarray):
        self.svm.fit(X, y)

    def predict(self, x):
        _x = x.reshape((1, x.shape[0]))
        self.svm.predict(x)

if __name__ == "__main__":
    dataloader = Loader(dataset_path)
    X_train, y_train, X_test, y_test = dataloader.load()
    svm = SVM()
    svm.train(X_train, y_train)

