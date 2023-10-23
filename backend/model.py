import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self) -> None:
    super(Model, self).__init__()
    self.fc1 = nn.Linear(28, 20)
    self.fc2 = nn.Linear(20, 10)
    self.fc3 = nn.Linear(10, 1)
  
  def forward(self, x) -> float:
    x = F.leaky_relu(self.fc1(x))
    x = F.leaky_relu(self.fc2(x))
    x = self.fc3(x)
    return x