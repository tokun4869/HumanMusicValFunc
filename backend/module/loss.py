import torch
import torch.nn as nn
import numpy as np
from module.io import load_model
from module.feature import musics2input
from module.const import *

class HumanMusicValLoss(nn.Module):
  def __init__(self, path: str) -> None:
    super().__init__()
    self.model = load_model(path)

  def forward(self, outputs):
    y = outputs.to("cpu").detach().numpy().copy()
    inputs = torch.Tensor(np.array(musics2input(y)))
    loss_tensor = -self.model(inputs)
    loss = torch.mean(loss_tensor)
    return loss
  
  def feature_forward(self, feature):
    inputs = torch.Tensor(feature)
    loss_tensor = -self.model(inputs)
    loss = torch.mean(loss_tensor)
    return loss
    