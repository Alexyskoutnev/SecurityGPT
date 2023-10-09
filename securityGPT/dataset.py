import sys
import os
import glob
from typing import List

import numpy as np
import pandas as pd

dataset_path = os.path.join("../data")
encoding = "utf-8"

def find_indices(text : str , char : str):
    indices = []
    for i, c in enumerate(text):
        if c == char:
            indices.append(i)
    return indices

class Loader(object):
    def __init__(self, dataset_path : str) -> None:
        self.dataset_path = dataset_path
        self._load()

    def parser(self, entries : List[str], dataset : str) -> np.ndarray:
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

    def _combine(self, entries : List[str]):
        return np.concatenate(entries)

    def _load(self):
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
            
    def filter(self):
        raise NotImplemented
    
    def split(self):
        raise NotImplemented
    
    def get(self):
        raise NotImplemented

if __name__ == "__main__":
    loader = Loader(dataset_path)