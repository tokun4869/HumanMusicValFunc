import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.features import music2represent, music2melspectrogram, music2feature
from module.exception import InvalidArgumentException
from module.const import *


# ===== ===== ===== =====
# Extractor
# ===== ===== ===== =====
class ReprExtractor(nn.Module):
  def __init__(self) -> None:
    super().__init__()
  
  def forward(self, inputs, required_func: bool = True) -> torch.Tensor: 
    if required_func:
      x = music2represent(inputs)
      return x
      # return torch.Tensor(np.array([music2represent(wave) for wave in inputs]))
    else:
      return inputs


class SpecExtractor(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    in_channels = [1, 2, 4]
    out_channels = [2, 4, 8]
    kernel_sizes = [3, 3, 3]
    strides = [1, 1, 1]
    paddings = [1, 1, 1]
    pool_sizes = [2, 2, 2]
    sequence_len = 3

    # 1 x 128 x 1292 -> 128 x 64 x 646 -> 4 x 32 x 323 -> 8 x 16 x 161 -> 20608
    self.sequence = nn.Sequential()
    for index in range(sequence_len):
      # self.sequence.add_module(f"BN2d_{index}", nn.BatchNorm2d(in_channels[index]))
      self.sequence.add_module(f"Conv2d_{index}", nn.Conv2d(in_channels[index], out_channels[index], kernel_size=kernel_sizes[index], stride=strides[index], padding=paddings[index]))
      self.sequence.add_module(f"ReLU_{index}", nn.ReLU())
      self.sequence.add_module(f"Pool_{index}", nn.MaxPool2d(pool_sizes[index]))
    self.sequence.add_module("Flatten", nn.Flatten())
    self.sequence.add_module("fc", nn.Linear(20608, 50))

  
  def forward(self, inputs: torch.Tensor, required_func: bool = True) -> torch.Tensor:
    if required_func:
      x = music2melspectrogram(inputs).unsqueeze(dim=1)
      x = torch.nn.functional.normalize(x, dim=-1)
    else:
      x = inputs.unsqueeze(dim=1)
    return self.sequence(x)
  

class CRNNExtractor(nn.Module):
  def __init__(self) -> None:
    super().__init__()
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

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    x = music2melspectrogram(inputs).unsqueeze(dim=1)
    x = self.cnn(x)
    x, h = self.rnn(x, None)
    x = self.fc(x)
    return x


class FeatExtractor(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    tempogram_channels = [[10, 20], [20, 30]]
    rms_channels = [[1, 2], [2, 4]]
    mfcc_channels = [[12, 24], [24, 48]]
    tonnetz_channels = [[6, 12], [12, 24]]
    zcr_channels = [[1, 2], [2, 4]]
    kernel_sizes = [3, 3]
    strides = [1, 1]
    paddings = [1, 1]
    pool_sizes = [2, 2]
    sequence_len = 2

    # 10 x 1292 -> 20 x 646 -> 30 x 323 -> 9690
    self.tempogram_seq = nn.Sequential()
    for index in range(sequence_len):
      self.tempogram_seq.add_module(f"tempogram_Conv1d_{index}", nn.Conv1d(tempogram_channels[0][index], tempogram_channels[1][index], kernel_size=kernel_sizes[index], stride=strides[index], padding=paddings[index]))
      # self.tempogram_seq.add_module(f"tempogram_BN1d_{index}", nn.BatchNorm1d(tempogram_channels[1][index]))
      self.tempogram_seq.add_module(f"tempogram_ReLU_{index}", nn.ReLU())
      self.tempogram_seq.add_module(f"tempogram_Pool_{index}", nn.MaxPool1d(pool_sizes[index]))
    self.tempogram_seq.add_module("tempogram_Flatten", nn.Flatten())
    self.tempogram_seq.add_module("tempogram_fc", nn.Linear(9690, 20))

    # 1 x 1292 -> 2 x 646 -> 4 x 323 -> 1292
    self.rms_seq = nn.Sequential()
    for index in range(sequence_len):
      self.rms_seq.add_module(f"rms_Conv1d_{index}", nn.Conv1d(rms_channels[0][index], rms_channels[1][index], kernel_size=kernel_sizes[index], stride=strides[index], padding=paddings[index]))
      # self.rms_seq.add_module(f"rms_BN1d_{index}", nn.BatchNorm1d(rms_channels[1][index]))
      self.rms_seq.add_module(f"rms_ReLU_{index}", nn.ReLU())
      self.rms_seq.add_module(f"rms_Pool_{index}", nn.MaxPool1d(pool_sizes[index]))
    self.rms_seq.add_module("rms_Flatten", nn.Flatten())
    self.rms_seq.add_module("rms_fc", nn.Linear(1292, 2))

    # 12 x 1292 -> 24 x 646 -> 48 x 323 -> 15504
    self.mfcc_seq = nn.Sequential()
    for index in range(sequence_len):
      self.mfcc_seq.add_module(f"mfcc_Conv1d_{index}", nn.Conv1d(mfcc_channels[0][index], mfcc_channels[1][index], kernel_size=kernel_sizes[index], stride=strides[index], padding=paddings[index]))
      # self.mfcc_seq.add_module(f"mfcc_BN1d_{index}", nn.BatchNorm1d(mfcc_channels[1][index]))
      self.mfcc_seq.add_module(f"mfcc_ReLU_{index}", nn.ReLU())
      self.mfcc_seq.add_module(f"mfcc_Pool_{index}", nn.MaxPool1d(pool_sizes[index]))
    self.mfcc_seq.add_module("mfcc_Flatten", nn.Flatten())
    self.mfcc_seq.add_module("mfcc_fc", nn.Linear(15504, 24))

    # 6 x 1292 -> 12 x 646 -> 24 x 323 -> 7752
    self.tonnetz_seq = nn.Sequential()
    for index in range(sequence_len):
      self.tonnetz_seq.add_module(f"tonnetz_Conv1d_{index}", nn.Conv1d(tonnetz_channels[0][index], tonnetz_channels[1][index], kernel_size=kernel_sizes[index], stride=strides[index], padding=paddings[index]))
      # self.tonnetz_seq.add_module(f"tonnetz_BN1d_{index}", nn.BatchNorm1d(tonnetz_channels[1][index]))
      self.tonnetz_seq.add_module(f"tonnetz_ReLU_{index}", nn.ReLU())
      self.tonnetz_seq.add_module(f"tonnetz_Pool_{index}", nn.MaxPool1d(pool_sizes[index]))
    self.tonnetz_seq.add_module("tonnetz_Flatten", nn.Flatten())
    self.tonnetz_seq.add_module("tonnetz_fc", nn.Linear(7752, 12))

    # 1 x 1292 -> 2 x 646 -> 4 x 323 -> 1292
    self.zcr_seq = nn.Sequential()
    for index in range(sequence_len):
      self.zcr_seq.add_module(f"zcr_Conv1d_{index}", nn.Conv1d(zcr_channels[0][index], zcr_channels[1][index], kernel_size=kernel_sizes[index], stride=strides[index], padding=paddings[index]))
      # self.zcr_seq.add_module(f"zcr_BN1d_{index}", nn.BatchNorm1d(zcr_channels[1][index]))
      self.zcr_seq.add_module(f"zcr_ReLU_{index}", nn.ReLU())
      self.zcr_seq.add_module(f"zcr_Pool_{index}", nn.MaxPool1d(pool_sizes[index]))
    self.zcr_seq.add_module("zcr_Flatten", nn.Flatten())
    self.zcr_seq.add_module("zcr_fc", nn.Linear(1292, 2))
  
  def forward(self, inputs, required_func: bool = True) -> torch.Tensor:
    if required_func:
      x = music2feature(inputs)
      x = torch.nn.functional.normalize(x, dim=-1)
    else:
      x = inputs
    tempogram = self.tempogram_seq(x[:, 0:10, :]) # 0 ~ 9
    rms = self.rms_seq(x[:, 10:11, :])            # 10
    mfcc = self.mfcc_seq(x[:, 11:23, :])          # 11 ~ 22
    tonnetz = self.tonnetz_seq(x[:, 23:29, :])    # 23 ~ 28
    zcr = self.zcr_seq(x[:, 29:30, :])            # 29
    return torch.cat((tempogram, rms, mfcc, tonnetz, zcr), dim=1)


class GRUExtractor(nn.Module):
  def __init__(self) -> None:
    super().__init__()

    tempogram_dim = 10
    rms_dim = 1
    mfcc_dim = 12
    centroid_dim = 1
    zcr_dim = 1

    self.tempogram_bn = nn.BatchNorm1d(tempogram_dim)
    self.tempogram_gru = nn.GRU(1292, 1292, batch_first=True)
    self.tempogram_flt = nn.Flatten()
    self.tempogram_fc = nn.Linear(1292, tempogram_dim * 2)

    self.rms_bn = nn.BatchNorm1d(rms_dim)
    self.rms_gru = nn.GRU(1292, 1292, batch_first=True)
    self.rms_flt = nn.Flatten()
    self.rms_fc = nn.Linear(1292, rms_dim * 2)

    self.mfcc_bn = nn.BatchNorm1d(mfcc_dim)
    self.mfcc_gru = nn.GRU(1292, 1292, batch_first=True)
    self.mfcc_flt = nn.Flatten()
    self.mfcc_fc = nn.Linear(1292, mfcc_dim * 2)

    self.centroid_bn = nn.BatchNorm1d(centroid_dim)
    self.centroid_gru = nn.GRU(1292, 1292, batch_first=True)
    self.centroid_flt = nn.Flatten()
    self.centroid_fc = nn.Linear(1292, centroid_dim * 2)

    self.zcr_bn = nn.BatchNorm1d(zcr_dim)
    self.zcr_gru = nn.GRU(1292, 1292, batch_first=True)
    self.zcr_flt = nn.Flatten()
    self.zcr_fc = nn.Linear(1292, zcr_dim * 2)
  
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
    x = music2feature(inputs)
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
    self.bn = nn.BatchNorm1d(50)
    self.fc1 = nn.Linear(50, 10)
    self.fc2 = nn.Linear(10, 1)
  
  def forward(self, x) -> float:
    x = F.leaky_relu(self.fc1(self.bn(x)))
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
  def __init__(self, extractor: str=EXTRACTOR_TYPE, head: str=HEAD_TYPE) -> None:
    super().__init__()
    self.extractor = self.__select_extractor(extractor)
    self.head = self.__select_head(head)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    z = self.extractor(x)
    out = self.head(z)
    return out

  def __select_extractor(self, extractor_type: str) -> ReprExtractor | SpecExtractor | FeatExtractor:
    if extractor_type == EXTRACTOR_REPR:
      return ReprExtractor()
    if extractor_type == EXTRACTOR_SPEC:
      return CRNNExtractor()
    if extractor_type == EXTRACTOR_FEAT:
      return GRUExtractor()
    if True:
      raise InvalidArgumentException(extractor_type)
  
  def __select_head(self, head_type: str):
    if head_type == HEAD_MLP:
      return HeadMLP()
    if head_type == HEAD_LR:
      return HeadLR()
    if True:
      raise InvalidArgumentException(head_type)