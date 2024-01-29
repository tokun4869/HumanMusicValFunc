import torch
import numpy as np
import module.features as f
from module.const import *

class TrainDataset(torch.utils.data.Dataset):
  def __init__(self, input_list: list, target_list: list[int], extractor, device: torch.device=torch.device("cpu")):
    input_tensor = torch.Tensor(np.array(input_list)).to(device)
    self.input = f.music2input(input_tensor, extractor=extractor, device=device)
    self.target = target_list
  
  def __len__(self):
    return len(self.target)
  
  def __getitem__(self, idx):
    return self.input[idx], self.target[idx]


class TestDataset(torch.utils.data.Dataset):
  def __init__(self, input_list: list):
    self.input = input_list
  
  def __len__(self):
    return len(self.input)
  
  def __getitem__(self, idx):
    return self.input[idx]


def standardization(feature_list: "list[np.ndarray[np.float32]]") -> "list[np.ndarray[np.float32]]":
  feature_norm_list = feature_list.copy()
  mean_list = np.mean(feature_list, axis=0)
  std_list = np.std(feature_list, axis=0)
  for i, feature in enumerate(feature_list):
    for j, one_feature in enumerate(feature):
      feature_norm_list[i][j] = (one_feature - mean_list[j]) / std_list[j] if 0 in std_list[j] else 0
  return feature_norm_list