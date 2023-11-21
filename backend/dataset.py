import torch

class Dataset(torch.utils.data.Dataset):
  def __init__(self, input_list: list, target_list: "list[int]"):
    self.input = input_list
    self.target = target_list
  
  def __len__(self):
    return len(self.target)
  
  def __getitem__(self, idx):
    return self.input[idx], self.target[idx]