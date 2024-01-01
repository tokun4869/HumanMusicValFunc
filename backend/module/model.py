import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.features import music2represent, music2melspectrogram, music2feature
from module.exception import InvalidArgumentException
from module.const import *


# ===== ===== ===== =====
# Normalize
# ===== ===== ===== =====
def normalize(x: torch.Tensor) -> torch.Tensor:
  return (x - torch.mean(x)) / (torch.sqrt(torch.var(x) + 1e-05))


# ===== ===== ===== =====
# Extractor
# ===== ===== ===== =====
class ReprExtractor(nn.Module):
  def __init__(self, device: torch.device=torch.device("cpu")) -> None:
    super().__init__()
    self.device = device
  
  def forward(self, x: torch.Tensor) -> torch.Tensor: 
    return music2represent(x, self.device)
  

class CRNNExtractor(nn.Module):
  def __init__(self, device: torch.device=torch.device("cpu")) -> None:
    super().__init__()
    self.device = device

    in_channels = [1, 30, 60, 60]
    out_channels = [30, 60, 60, 60]
    kernel_sizes = [3, 3, 3, 3]
    strides = [1, 1, 1, 1]
    paddings = [1, 1, 1, 1]
    pool_sizes = [(2, 2), (3, 3), (4, 4), (4, 4)]
    sequence_len = 4

    # 1 x 128 x 1292 -> 30 x 64 x 646 -> 60 x 21 x 215 -> 60 x 5 x 53 -> 60 x 1 x 13 -> 780
    self.cnn = nn.Sequential()
    for index in range(sequence_len):
      self.cnn.add_module(f"BN2d_{index}", nn.BatchNorm2d(in_channels[index]))
      self.cnn.add_module(f"Conv2d_{index}", nn.Conv2d(in_channels[index], out_channels[index], kernel_size=kernel_sizes[index], stride=strides[index], padding=paddings[index]))
      self.cnn.add_module(f"ReLU_{index}", nn.ReLU())
      self.cnn.add_module(f"Pool_{index}", nn.MaxPool2d(pool_sizes[index]))
    self.cnn.add_module("Flatten", nn.Flatten())
    
    self.rnn = nn.GRU(780, 780, 2)

    self.fc = nn.Linear(780, 50)

    self.cnn = self.cnn.to(device)
    self.rnn = self.rnn.to(device)
    self.fc = self.fc.to(device)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    x = music2melspectrogram(inputs, self.device).unsqueeze(dim=1)
    x = self.cnn(x)
    x, h = self.rnn(x, None)
    x = self.fc(x)
    return x


class GRUExtractor(nn.Module):
  def __init__(self, device: torch.device=torch.device("cpu")) -> None:
    super().__init__()
    self.device = device

    tempogram_dim = 10
    rms_dim = 1
    mfcc_dim = 12
    centroid_dim = 1
    zcr_dim = 1

    self.tempogram_bn = nn.BatchNorm1d(tempogram_dim).to(device)
    self.tempogram_gru = nn.GRU(1292, 1292, batch_first=True).to(device)
    self.tempogram_flt = nn.Flatten().to(device)
    self.tempogram_fc = nn.Linear(1292, tempogram_dim * 2).to(device)

    self.rms_bn = nn.BatchNorm1d(rms_dim).to(device)
    self.rms_gru = nn.GRU(1292, 1292, batch_first=True).to(device)
    self.rms_flt = nn.Flatten().to(device)
    self.rms_fc = nn.Linear(1292, rms_dim * 2).to(device)

    self.mfcc_bn = nn.BatchNorm1d(mfcc_dim).to(device)
    self.mfcc_gru = nn.GRU(1292, 1292, batch_first=True).to(device)
    self.mfcc_flt = nn.Flatten().to(device)
    self.mfcc_fc = nn.Linear(1292, mfcc_dim * 2).to(device)

    self.centroid_bn = nn.BatchNorm1d(centroid_dim).to(device)
    self.centroid_gru = nn.GRU(1292, 1292, batch_first=True).to(device)
    self.centroid_flt = nn.Flatten().to(device)
    self.centroid_fc = nn.Linear(1292, centroid_dim * 2).to(device)

    self.zcr_bn = nn.BatchNorm1d(zcr_dim).to(device)
    self.zcr_gru = nn.GRU(1292, 1292, batch_first=True).to(device)
    self.zcr_flt = nn.Flatten().to(device)
    self.zcr_fc = nn.Linear(1292, zcr_dim * 2).to(device)
  
  def tempogram_forward(self, x: torch.Tensor) -> torch.Tensor:
    tempogram = self.tempogram_bn(x)
    _, h = self.tempogram_gru(tempogram, None)
    h = self.tempogram_flt(h.transpose(0, 1))
    return self.tempogram_fc(h)
  
  def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
    rms = self.rms_bn(x)
    _, h = self.rms_gru(rms, None)
    h = self.rms_flt(h.transpose(0, 1))
    return self.rms_fc(h)
  
  def mfcc_forward(self, x: torch.Tensor) -> torch.Tensor:
    mfcc = self.mfcc_bn(x)
    _, h = self.mfcc_gru(mfcc, None)
    h = self.mfcc_flt(h.transpose(0, 1))
    return self.mfcc_fc(h)
  
  def centroid_forward(self, x: torch.Tensor) -> torch.Tensor:
    centroid = self.centroid_bn(x)
    _, h = self.centroid_gru(centroid, None)
    h = self.centroid_flt(h.transpose(0, 1))
    return self.centroid_fc(h)
  
  def zcr_forward(self, x: torch.Tensor) -> torch.Tensor:
    zcr = self.zcr_bn(x)
    _, h = self.zcr_gru(zcr, None)
    h = self.zcr_flt(h.transpose(0, 1))
    return self.zcr_fc(h)
  
  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    x = music2feature(inputs, self.device)
    tempogram = self.tempogram_forward(x[:, 0:10, :]) # 0 ~ 9
    rms = self.rms_forward(x[:, 10:11, :])            # 10
    mfcc = self.mfcc_forward(x[:, 11:23, :])          # 11 ~ 22       
    centroid = self.centroid_forward(x[:, 23:24, :])  # 23
    zcr = self.zcr_forward(x[:, 24:25, :])            # 24
    return torch.cat((tempogram, rms, mfcc, centroid, zcr), dim=1)


# ===== ===== ===== =====
# Head
# ===== ===== ===== =====
class HeadMLP(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.fc1 = nn.Linear(50, 10)
    self.fc2 = nn.Linear(10, 1)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = normalize(x)
    x = F.leaky_relu(self.fc1(x))
    x = self.fc2(x)
    return x


class HeadLR(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.linear = nn.Linear(60, 1)
  
  def forward(self, x) -> float:
    return self.linear(x)


# ===== ===== ===== =====
# Extractor + Head
# ===== ===== ===== =====
class Model(nn.Module):
  def __init__(self, extractor: str=EXTRACTOR_TYPE, head: str=HEAD_TYPE, device: torch.device=torch.device("cpu")) -> None:
    super().__init__()
    self.extractor = self.__select_extractor(extractor, device)
    self.extractor = self.extractor.to(device)
    self.head = self.__select_head(head)
    self.head = self.head.to(device)
    self.device = device
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    z = self.extractor(x)
    out = self.head(z)
    return out

  def __select_extractor(self, extractor_type: str, device: torch.device=torch.device("cpu")) -> ReprExtractor | CRNNExtractor | GRUExtractor:
    if extractor_type == EXTRACTOR_REPR:
      return ReprExtractor(device)
    if extractor_type == EXTRACTOR_SPEC:
      return CRNNExtractor(device)
    if extractor_type == EXTRACTOR_FEAT:
      return GRUExtractor(device)
    if True:
      raise InvalidArgumentException(extractor_type)
  
  def __select_head(self, head_type: str):
    if head_type == HEAD_MLP:
      return HeadMLP()
    if head_type == HEAD_LR:
      return HeadLR()
    if True:
      raise InvalidArgumentException(head_type)