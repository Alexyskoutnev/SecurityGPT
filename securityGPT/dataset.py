import sys
import os
import glob
from typing import Optional, Tuple, Union, List, Generator

import numpy as np
import pandas as pd
import nltk
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec, Doc2Vec
import gensim

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
    def __init__(self, dataset_path : str, train_size: float = 0.8,
                size: Optional[int] = 0, torch : bool = False, 
                word2vec : bool = False, tfidf : bool = False,
                doc2vec : bool = False, batch_size : int = 32, 
                padding_bool : bool = True) -> None:
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
        self.torch = torch
        self.word2vec = word2vec
        self.tfidf = tfidf
        self.doc2vec = doc2vec
        self.embedding_dim = 200
        self.max_sentence_length = 200
        self.embedding_model = None
        self.batch_size = batch_size
        self.padding_bool = padding_bool
        self.raw = False
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

    def _sentence_embedding(self, tokenized_sentence : list[str]) -> np.ndarray:
        """
        Generate a sentence embedding using the underlying Doc2Vec model.

        Parameters:
        - tokenized_sentence (list[str]): A list of tokenized words in the sentence.

        Returns:
        - np.ndarray: A numpy array representing the sentence embedding.
        Each element in the array corresponds to the embedding vector for a token in the sentence.
        The resulting array has a shape of (len(tokenized_sentence), embedding_dimension).

        Note:
        - This function assumes that the `embedding_model` attribute is an instance of Doc2Vec,
        and the `infer_vector` method is available for generating word embeddings.
        Make sure that the model has been appropriately trained before calling this function.
        """
        return np.array([self.embedding_model.infer_vector(doc) for doc in tokenized_sentence], dtype=np.float32)

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

        Reads CSV files in the specified dataset path, extracts text and labels,
        and combines the data. If word embeddings are enabled, it optionally trains
        a Word2Vec model on the preprocessed text data.

        Returns:
            None: The function stores the processed data and labels in self.X and self.y.

        Notes:
            - The function assumes that CSV files contain text data labeled with "Chromium" or "OpenStack."
            - The function uses a parser specified during object initialization to extract text and labels.
            - The preprocessing step involves filtering and processing the extracted text.
            - If word embeddings are enabled, the function trains a Word2Vec model on the preprocessed text data.

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
        if self.word2vec and self.padding_bool:
            _text_temp_split = [sentence.split(" ") for sentence in _text_temp]
            self.embedding_model = Word2Vec(_text_temp_split, vector_size=self.embedding_dim, window=5, min_count=1, sg=0)
            self.X = self._vectorize_embeddings(_text_temp_split)
            self.y = np.array(_label_temp)
        elif self.doc2vec:
            _text_temp_tokens = [word_tokenize(sentence.lower()) for sentence in _text_temp]
            tagged_data = [gensim.models.doc2vec.TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(_text_temp_tokens)]
            self.embedding_model = gensim.models.doc2vec.Doc2Vec(vector_size=self.embedding_dim, min_count=2, epochs=40)
            self.embedding_model.build_vocab(tagged_data)
            self.X = self._sentence_embedding(_text_temp_tokens)
            self.y = np.array(_label_temp)
        elif self.tfidf:
            self.X = self._vectorize(_text_temp)
            self.y = np.array(_label_temp)
        else:
            self.X = _text_temp
            self.y = np.array(_label_temp)
       


    def _padding(self, sentence : list) -> np.array:
        """
        Pad or truncate a sentence to a fixed length.

        Parameters:
        - sentence (np.array): Array of word vectors for a sentence.

        Returns:
        - Padded or truncated sentence with shape (max_sentence_length, embedding_dim).
        """
        _padded_data = np.zeros((self.max_sentence_length, self.embedding_dim))
        _data = np.array(sentence[:self.max_sentence_length])
        seq_length, emb_dim = _data.shape
        _padded_data[:seq_length, :] = _data
        return _padded_data

    def _vectorize_embeddings(self, data : list[str] ) -> np.array:
        """
        Vectorize a list of sentences using Word2Vec embeddings to sequence-based 
        encoding.

        Parameters:
        - data (List[str]): List of input sentences. 
          For example [['Testing', 'chromium', 'id', 'works', '2', 'problem', '1', '2', '3'], [...]]

        Returns:
        - Vectorized sentences as a NumPy array with padding/truncation 
          as size [dataset_size, length_of_max_sentence_embedding, word_vec_dim].
        """
        vectorized_sentences = []
        for sentence in data:
            sentence_vectors = []
            for word in sentence:
                if word in self.embedding_model.wv:
                    word_vector = self.embedding_model.wv[word]
                    sentence_vectors.append(word_vector)
            padded_sentence_vector = self._padding(sentence_vectors)
            vectorized_sentences.append(padded_sentence_vector)
        return np.array(vectorized_sentences, dtype=np.float32)

    def random_oversampler(self, X : np.ndarray, 
                                 y : np.ndarray, 
                                 target_label : int  = 1, 
                                 ratio : float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
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
        if self.tfidf:
            X = X.toarray()
        n_samples = y.shape[0]
        target_indices = np.where(y == target_label)[0]
        num_to_sample = int(ratio * n_samples) - target_indices.shape[0]
        sampled_indices = np.random.choice(target_indices, size=num_to_sample, replace=True)
        X_resampled = np.vstack((X, X[sampled_indices]))
        y_resampled = np.hstack((y, y[sampled_indices]))
        return X_resampled, y_resampled
    
    def _batch_load(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate batches of feature data and corresponding labels from the internal training dataset.

        Yields:
        - Tuple[np.ndarray, np.ndarray]: A tuple containing a batch of feature data and its corresponding labels.

        This method splits the internal training data (_X_train and _y_train) into smaller batches,
        and it yields each batch as a tuple. It is useful for processing large datasets in smaller chunks,
        which can reduce memory usage and improve efficiency.

        Yields:
            Tuple[np.ndarray, np.ndarray]: A tuple containing a batch of feature data and its corresponding labels.
            
        Raises:
            ValueError: If the specified batch size is less than or equal to 0.

        """
        if self.batch_size <= 0:
            raise ValueError("The batch parameter should be greater than zero.")
        for i in range(0, self._X_train.shape[0], self.batch_size):
            if self.torch:
                yield torch.from_numpy(self._X_train[i:i + self.batch_size]), torch.from_numpy(self._y_train[i:i + self.batch_size])
            else:
                yield self._X_train[i:i + self.batch_size], self._y_train[i:i + self.batch_size]


    def load(self, seed : Optional[int] = None, 
                   bootstrap : bool = False, 
                   ratio : float = 0.1, 
                   batches : int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and split the data into training and testing sets.

        Parameters:
            seed (Optional[int]): Seed for the random number generator (optional).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing X_train, X_test, y_train, y_test.
        """
        _X_train, _X_test, _y_train, _y_test = self._split(self.train_size, seed, bootstrap, ratio)
        self._X_train = _X_train.astype(np.float32)
        self._y_train = _y_train
        if not self.embedding_model and not self.tfidf:
            self._X_train = self._X_train[:, np.newaxis, :]
            _X_test = _X_test[:, np.newaxis, :]
        if self.torch:
            _X_test = torch.from_numpy(_X_test.astype(np.float32))
            _y_test = torch.from_numpy(_y_test)
        data = self._batch_load
        return self._X_train, self._y_train, _X_test, _y_test, data

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

class DataTorch(Dataset):
    def __init__(self, dataset_path : str, train_size: float = 0.8,
                size: Optional[int] = 0, torch : bool = False, 
                batch_size : int = 32, use_tokenizer : bool = False,
                seed : Optional[Union[None, int]] = None, bootstrap : bool = True,
                ratio : float = 0.2, test_data : bool = False, 
                train_data : bool = True) -> None:
        """
        Initialize the Loader object.

        Parameters:
            dataset_path (str): The path to the dataset.
            train_size (float): The proportion of the data to include in the training split (default is 0.8).
        """
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.dataset_path = dataset_path
        self.train_size = train_size
        self.size = size
        self.torch = torch
        self.embedding_dim = 200
        self.max_sentence_length = 200
        self.batch_size = batch_size
        self.seed = 0 if seed is not None else seed
        self.bootstrap = bootstrap
        self.ratio = ratio
        self.test_data = None if test_data is None else test_data
        self.train_data = None if train_data is None else train_data
        self._load()

    def __len__(self):
        return self._size

    def __getitem__(self, item : int):
        return {
            'text' : self.X[item],
            'label' : str(self.y[item])
        }

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
        if self.bootstrap:
            X, y = self.random_oversampler(X, y, ratio=ratio)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=random_seed)
        return X_train, X_test, y_train, y_test

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

    def _load(self):
        """
        Load and combine data from CSV files in the dataset path.

        Reads CSV files in the specified dataset path, extracts text and labels,
        and combines the data. If word embeddings are enabled, it optionally trains
        a Word2Vec model on the preprocessed text data.

        Returns:
            None: The function stores the processed data and labels in self.X and self.y.

        Notes:
            - The function assumes that CSV files contain text data labeled with "Chromium" or "OpenStack."
            - The function uses a parser specified during object initialization to extract text and labels.
            - The preprocessing step involves filtering and processing the extracted text.
            - If word embeddings are enabled, the function trains a Word2Vec model on the preprocessed text data.

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
        self.X = np.array(_text_temp)
        self.y = np.array(_label_temp)
        _X_train, _X_test, _y_train, _y_test = self._split(self.train_size, self.seed, self.bootstrap, self.ratio)
        self._size = len(_y_train)
        if self.train_data:
            self.X = _X_train
            self.y = _y_train
        elif self.test_data:
            self.X = _X_test
            self.y = _y_test


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

    def random_oversampler(self, X : np.ndarray, y : np.ndarray, target_label : int  = 1, ratio : float = 0.2) -> Tuple[list, list]:
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
        n_samples = y.shape[0]
        target_indices_1 = np.where(y == target_label)[0]
        target_indices_0 = np.where(y == 0)[0]
        num_to_sample = int(ratio * n_samples) - target_indices_1.shape[0]
        num_to_add_orignal = len(X) - num_to_sample
        X_resampled = []
        y_resampled = []
        sampled_indices_1 = np.random.choice(target_indices_1, size=num_to_sample, replace=True)
        sampled_indices_0 = np.random.choice(target_indices_0, size=num_to_add_orignal, replace=True)
        for i in sampled_indices_1:
            X_resampled.append(X[i])
            y_resampled.append(y[i])
        for i in sampled_indices_0:
            X_resampled.append(X[i])
            y_resampled.append(y[i])
        return X_resampled, y_resampled

if __name__ == "__main__":
    size = 1000 # use tiny dataset to save time debugging
    loader = DataTorch(dataset_path, size=size)