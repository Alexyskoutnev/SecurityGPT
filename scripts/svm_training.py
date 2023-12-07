import sys
import os

from securityGPT.dataset import Loader
from securityGPT.algorithms.svm import SVM
from typing import List, Tuple, Generator
from datetime import datetime
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

import numpy as np
import joblib

dataset_path = os.path.join("./data")
dataset_folder = os.path.join("./models/svm")

def main():
    size = 1000
    ratio = 0.2
    SGD = True #Use Gradient Descent (better for larger scale datasets)
    dataloader = Loader(dataset_path, size=size)
    X_train, y_train, X_test, y_test, data = dataloader.load(bootstrap=True, ratio=ratio)
    kernel = 'linear'
    svm = SVM(kernel=kernel, SGD=SGD)
    svm.train(data)
    X_test = np.squeeze(X_test, axis=1)
    breakpoint()
    y_pred = svm.predict(X_test)
    acc, f1 = svm.stats(y_test, y_pred)  # Corrected here
    svm.save()
    print(f"Accuracy: [{acc}] \nF1 Score: [{f1}]")

if __name__ == "__main__":
    main()