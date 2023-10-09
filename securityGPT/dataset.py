import sys
import os
import glob
from typing import Optional, Tuple, Union, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_path = os.path.join("../data")
encoding = "utf-8"

def find_indices(text : str , char : str) -> List[int]:
    """
    Find the indices of a character in a string.

    Parameters:
        text (str): The input text.
        char (str): The character to search for.

    Returns:
        List[int]: A list of indices where the character is found in the input text.
    """
    indices = []
    for i, c in enumerate(text):
        if c == char:
            indices.append(i)
    return indices

class Loader(object):
    def __init__(self, dataset_path : str, train_size: float = 0.8) -> None:
        """
        Initialize the Loader object.

        Parameters:
            dataset_path (str): The path to the dataset.
            train_size (float): The proportion of the data to include in the training split (default is 0.8).
        """
        self.dataset_path = dataset_path
        self.train_size = train_size
        self._load()

    def parser(self, entries : List[str], dataset : str) -> np.ndarray:
        """
        Parse the dataset entries and return a NumPy array.

        Parameters:
            entries (List[str]): List of dataset entries.
            dataset (str): The dataset name ("Chromium" or "OpenStack").

        Returns:
            np.ndarray: A NumPy array containing parsed data.
        """
        entries = entries[1:] #skipping headers
        data = list()
        if dataset == "Chromium":
            for entry in entries:
                if len(entry) == 0:
                    continue
                try:
                    colon_idx = entry.index(":")
                    comma_idxs = find_indices(entry, ",")
                    second_two_last_comma_idx = comma_idxs[-2]
                    text = entry[colon_idx+1:second_two_last_comma_idx]
                    security_bool = int(entry[comma_idxs[-2]+1:comma_idxs[-1]])
                    data_entry = [text.strip(), security_bool]
                    data.append(data_entry)
                except:
                    print(f"skipping entry -> [{entry[:1]}]")
        elif dataset == "OpenStack":
            for entry in entries:
                if len(entry) == 0:
                    continue
                try:
                    comma_idxs = find_indices(entry, ",")
                    text = entry[comma_idxs[0]+1:comma_idxs[-1]]
                    security_bool = int(entry[comma_idxs[-1]+1:])
                    data_entry = [text.strip(), security_bool]
                    data.append(data_entry)
                except:
                    print(f"skipping entry -> [{entry[:1]}]")
        return data

    def _combine(self, entries : List[str]) -> np.ndarray:
        """
        Combine a list of NumPy arrays into a single NumPy array.

        Parameters:
            entries (List[np.ndarray]): List of NumPy arrays to combine.

        Returns:
            np.ndarray: A single NumPy array containing combined data.
        """
        return np.concatenate(entries)

    def _load(self):
        """
        Load and combine data from CSV files in the dataset path.
        """
        csv_files = glob.glob(os.path.join(self.dataset_path, '*.csv'))
        data_list = list()
        for csv in csv_files:
            if "Chromium" in csv:
                with open(csv, 'rb') as f:
                    file_bytes = f.read()
                    file_text = file_bytes.decode("utf-8", errors="ignore")
                    data_list.append(self.parser(file_text.split("\n"), "Chromium"))
            elif "OpenStack" in csv:
                with open(csv, 'rb') as f:
                    file_bytes = f.read()
                    file_text = file_bytes.decode("utf-8", errors="ignore")
                    data_list.append(self.parser(file_text.split("\n"), "OpenStack"))
        self.data = self._combine(data_list)

    def load(self, seed : Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and split the data into training and testing sets.

        Parameters:
            seed (Optional[int]): Seed for the random number generator (optional).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing X_train, X_test, y_train, y_test.
        """
        X_train, y_train, X_test, y_test = self._split(self.train_size, seed)
        return X_train, y_train, X_test, y_test

    def filter(self) -> None:
        raise NotImplemented
    
    def _split(self, train_size: float = 0.8, random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the data into training and testing sets.

        Parameters:
            train_size (float): The proportion of the data to include in the training split (default is 0.8).
            random_seed (Optional[int]): Seed for the random number generator (optional).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing X_train, X_test, y_train, y_test.
        """
        X = self.data[:, :-1]
        y = self.data[:, -1].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=random_seed)
        return X_train, X_test, y_train, y_test
    
    def get(self, idx : int) -> np.ndarray:
        """
        Get a data entry by index.

        Parameters:
            idx (int): Index of the data entry to retrieve.

        Returns:
            np.ndarray: The data entry at the specified index.
        """
        return self.data[idx]

if __name__ == "__main__":
    loader = Loader(dataset_path)
    data = loader.load()