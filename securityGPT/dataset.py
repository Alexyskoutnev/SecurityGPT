import sys
import os
import glob
from typing import Optional, Tuple, Union, List, Generator

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

dataset_path = os.path.join("../data")
encoding = "utf-8"
nltk.download('punkt')

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

class LoaderTest(object):
    """Loader for Shakespeare Dataset

    Args:
        object (_type_): _description_
    """
    pass

class Loader(object):
    def __init__(self, dataset_path : str, train_size: float = 0.8, size: Optional[int] = 0) -> None:
        """
        Initialize the Loader object.

        Parameters:
            dataset_path (str): The path to the dataset.
            train_size (float): The proportion of the data to include in the training split (default is 0.8).
        """
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer()
        self.dataset_path = dataset_path
        self.train_size = train_size
        self.size = size
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

    def _combine(self, X : np.ndarray, y : np.ndarray) -> np.ndarray:
        return None

    def _vectorize(self, text : List[str]) -> np.ndarray:
        """
        Combine a list of NumPy arrays into a single NumPy array.

        Parameters:
            entries (List[np.ndarray]): List of NumPy arrays to combine.

        Returns:
            np.ndarray: A single NumPy array containing combined data.
        """
        X = self.tfidf_vectorizer.fit_transform(text)
        return X

    def _load(self):
        """
        Load and combine data from CSV files in the dataset path.
        """
        csv_files = glob.glob(os.path.join(self.dataset_path, '*.csv'))
        data_list = list()
        _text_temp = list()
        _label_temp = list()
        for csv in csv_files:
            if "Chromium" in csv:
                with open(csv, 'rb') as f:
                    file_bytes = f.read()
                    file_text = file_bytes.decode("utf-8", errors="ignore")
                    if self.size > 0:
                        parsed_text = self.parser(file_text.split("\n"), "Chromium")[:self.size]
                    else:
                        parsed_text = self.parser(file_text.split("\n"), "Chromium")
                    for text, label in parsed_text:
                        filtered_text = self._process(text)
                        _text_temp.append(filtered_text)
                        _label_temp.append(label)
            elif "OpenStack" in csv:
                with open(csv, 'rb') as f:
                    file_bytes = f.read()
                    file_text = file_bytes.decode("utf-8", errors="ignore")
                    if self.size > 0:
                        parsed_text = self.parser(file_text.split("\n"), "OpenStack")[:self.size]
                    else:
                        parsed_text = self.parser(file_text.split("\n"), "OpenStack")
                    for text, label in parsed_text:
                        filtered_text = self._process(text)
                        _text_temp.append(filtered_text)
                        _label_temp.append(label)
        self.X = self._vectorize(_text_temp)
        self.y = np.array(_label_temp)

    def random_oversampler(self, X : np.ndarray, y : np.ndarray, target_label : int  = 1, ratio : float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly oversamples the minority class in a dataset.

        Parameters:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The label vector.
            target_label (int, optional): The label to oversample (default is 1).
            ratio (float, optional): The desired ratio of the minority class after oversampling (default is 0.1).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the oversampled feature matrix and label vector.
        """
        X = X.toarray()
        n_samples = y.shape[0]
        target_indices = np.where(y == target_label)[0]
        num_to_sample = int(ratio * n_samples) - target_indices.shape[0]
        sampled_indices = np.random.choice(target_indices, size=num_to_sample, replace=True)
        X_resampled = np.vstack((X, X[sampled_indices]))
        y_resampled = np.hstack((y, y[sampled_indices]))
        return X_resampled, y_resampled
    
    def _batch_load(self, batches : int = 10) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate batches of feature data and corresponding labels from the internal training dataset.

        Parameters:
        - batches (int, optional): The number of batches to generate (default is 8).

        Yields:
        - Tuple[np.ndarray, np.ndarray]: A tuple containing a batch of feature data and its corresponding labels.

        This method splits the internal training data (_X_train and _y_train) into smaller batches, and it yields each batch as a tuple. It is useful for processing large datasets in smaller chunks, which can reduce memory usage and improve efficiency.

        Examples:
        ```
        for X_batch, y_batch in self._batch_load(batches=4):
            # Process the batch of data and labels here
        ```
        """
        batch_size = self._X_train.shape[0] // batches
        for i in range(0, self._X_train.shape[0], batch_size):
            yield self._X_train[i:i + batch_size], self._y_train[i:i + batch_size]

    def load(self, seed : Optional[int] = None, bootstrap : bool = False, ratio : float = 0.1, batches : int = 17) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and split the data into training and testing sets.

        Parameters:
            seed (Optional[int]): Seed for the random number generator (optional).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing X_train, X_test, y_train, y_test.
        """
        _X_train, _X_test, _y_train, _y_test = self._split(self.train_size, seed, bootstrap, ratio)
        self._X_train = _X_train
        self._y_train = _y_train
        self.batches = batches
        data = self._batch_load
        return _X_train, _y_train, _X_test, _y_test, data

    def _process(self, text : str) -> str:
        """
        Preprocesses the input text by tokenizing, removing non-alphanumeric tokens, and stopwords.

        Parameters:
        - text (str): The input text to be preprocessed.

        Returns:
        - str: The preprocessed text after tokenization, alphanumeric filtering, and stopword removal.
        """
        tokens = nltk.word_tokenize(text)
        filter_tokens = list()
        for i, word in enumerate(tokens):
            if word.isalnum() and word not in self.stop_words:
                filter_tokens.append(word)
            else:
                continue
        return ' '.join(filter_tokens)
    
    def _split(self, train_size: float = 0.8, random_seed: Optional[int] = None, bootstrap : bool = False, ratio : float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the data into training and testing sets.

        Parameters:
            train_size (float): The proportion of the data to include in the training split (default is 0.8).
            random_seed (Optional[int]): Seed for the random number generator (optional).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing X_train, X_test, y_train, y_test.
        """
        X = self.X
        y = self.y.astype(int)
        if bootstrap:
            X, y = self.random_oversampler(X, y, ratio=ratio)
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
    size = 1000 # use tiny dataset to save time debugging
    loader = Loader(dataset_path, size=size)
    data = loader.load(bootstrap=True)