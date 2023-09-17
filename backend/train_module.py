import numpy as np
import random

def dataset_load(train_data_dir: str, target_data_dir: str) -> "tuple[np.ndarray[np.float32], np.ndarray[np.float32]]":
  train = np.load(train_data_dir)
  target = np.load(target_data_dir)
  return train, target

def valid_index(array: "np.ndarray[np.float32]", n_index: int):
  index_list = []
  while len(index_list) < n_index:
    index = random.randint(0, array.shape[0])
    if index not in index_list:
      index_list.append(index)
  return index_list