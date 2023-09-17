import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self) -> None:
    super(Model, self).__init__()
    self.fc1 = nn.Linear(44, 10)
    self.fc2 = nn.Linear(10, 1)
  
  def forward(self, x) -> float:
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x