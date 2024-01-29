import torch
import torch.nn as nn
from module.io import load_model
from module.const import *

class HumanMusicValLoss(nn.Module):
  def __init__(self, path: str, extractor: str=EXTRACTOR_TYPE, device: torch.device=torch.device("cpu")) -> None:
    super().__init__()
    self.model = load_model(path, extractor=extractor, device=device)

  def forward(self, outputs: torch.Tensor):
    loss_tensor = -self.model(outputs)
    loss = torch.sum(loss_tensor)
    return loss
  
    