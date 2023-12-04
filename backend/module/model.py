import torch
import torch.nn as nn
import torch.nn.functional as F
from module.const import *

class ReprMPL(nn.Module):
  def __init__(self) -> None:
    super(ReprMPL, self).__init__()
    self.fc1 = nn.Linear(60, 10)
    self.fc2 = nn.Linear(10, 1)
  
  def forward(self, x) -> float:
    x = F.leaky_relu(self.fc1(x))
    x = self.fc2(x)
    return x


class ReprLinearRegression(nn.Module):
  def __init__(self) -> None:
    super(ReprLinearRegression, self).__init__()
    self.linear = nn.Linear(60, 1)
  
  def forward(self, x) -> float:
    return self.linear(x)


class SpecCNN(nn.Module):
  def __init__(self) -> None:
    super(SpecCNN, self).__init__()
    in_channels = [1, 2, 4]
    out_channels = [2, 4, 8]
    kernel_sizes = [3, 3, 3]
    strides = [1, 1, 1]
    paddings = [1, 1, 1]
    pool_sizes = [2, 2, 2]
    sequence_len = 3

    # 1 x 128 x 1292 -> 2 x 64 x 646 -> 4 x 32 x 323 -> 8 x 16 x 161 -> 20608
    self.sequence = nn.Sequential()
    for index in range(sequence_len):
      self.sequence.add_module(f"Conv2d_{index}", nn.Conv2d(in_channels[index], out_channels[index], kernel_size=kernel_sizes[index], stride=strides[index], padding=paddings[index]))
      self.sequence.add_module(f"BN2d_{index}", nn.BatchNorm2d(out_channels[index]))
      self.sequence.add_module(f"LeakyReLU_{index}", nn.LeakyReLU())
      self.sequence.add_module(f"Pool_{index}", nn.MaxPool2d(pool_sizes[index]))
    self.sequence.add_module("Flatten", nn.Flatten())
    self.sequence.add_module("fc", nn.Linear(20608, 1))

  def forward(self, x) -> float:
    x = self.sequence(x)
    return x


class FeatCNN(nn.Module):
  def __init__(self) -> None:
    super(FeatCNN, self).__init__()
    tempogram_channels = [[10, 20, 30], [20, 30, 40]]
    rms_channels = [[1, 2, 4], [2, 4, 8]]
    mfcc_channels = [[12, 24, 48], [24, 48, 96]]
    tonnetz_channels = [[6, 12, 24], [12, 24, 48]]
    zcr_channels = [[1, 2, 4], [2, 4, 8]]
    kernel_sizes = [3, 3, 3]
    strides = [1, 1, 1]
    paddings = [1, 1, 1]
    pool_sizes = [2, 2]
    sequence_len = 2

    # 10 x 1292 -> 20 x 646 -> 30 x 323 -> 9690
    self.tempogram_seq = nn.Sequential()
    for index in range(sequence_len):
      self.tempogram_seq.add_module(f"tempogram_Conv1d_{index}", nn.Conv1d(tempogram_channels[0][index], tempogram_channels[1][index], kernel_size=kernel_sizes[index], stride=strides[index], padding=paddings[index]))
      self.tempogram_seq.add_module(f"tempogram_BN1d_{index}", nn.BatchNorm1d(tempogram_channels[1][index]))
      self.tempogram_seq.add_module(f"tempogram_LeakyReLU_{index}", nn.LeakyReLU())
      self.tempogram_seq.add_module(f"tempogram_Pool_{index}", nn.MaxPool1d(pool_sizes[index]))
    self.tempogram_seq.add_module("tempogram_Flatten", nn.Flatten())
    self.tempogram_seq.add_module("tempogram_fc", nn.Linear(9690, 10))
    self.tempogram_seq.add_module("tempogram_output", nn.LeakyReLU())

    # 1 x 1292 -> 2 x 646 -> 4 x 323 -> 1292
    self.rms_seq = nn.Sequential()
    for index in range(sequence_len):
      self.rms_seq.add_module(f"rms_Conv1d_{index}", nn.Conv1d(rms_channels[0][index], rms_channels[1][index], kernel_size=kernel_sizes[index], stride=strides[index], padding=paddings[index]))
      self.rms_seq.add_module(f"rms_BN1d_{index}", nn.BatchNorm1d(rms_channels[1][index]))
      self.rms_seq.add_module(f"rms_LeakyReLU_{index}", nn.LeakyReLU())
      self.rms_seq.add_module(f"rms_Pool_{index}", nn.MaxPool1d(pool_sizes[index]))
    self.rms_seq.add_module("rms_Flatten", nn.Flatten())
    self.rms_seq.add_module("rms_fc", nn.Linear(1292, 1))
    self.rms_seq.add_module("rms_output", nn.LeakyReLU())

    # 12 x 1292 -> 24 x 646 -> 48 x 323 -> 15504
    self.mfcc_seq = nn.Sequential()
    for index in range(sequence_len):
      self.mfcc_seq.add_module(f"mfcc_Conv1d_{index}", nn.Conv1d(mfcc_channels[0][index], mfcc_channels[1][index], kernel_size=kernel_sizes[index], stride=strides[index], padding=paddings[index]))
      self.mfcc_seq.add_module(f"mfcc_BN1d_{index}", nn.BatchNorm1d(mfcc_channels[1][index]))
      self.mfcc_seq.add_module(f"mfcc_LeakyReLU_{index}", nn.LeakyReLU())
      self.mfcc_seq.add_module(f"mfcc_Pool_{index}", nn.MaxPool1d(pool_sizes[index]))
    self.mfcc_seq.add_module("mfcc_Flatten", nn.Flatten())
    self.mfcc_seq.add_module("mfcc_fc", nn.Linear(15504, 12))
    self.mfcc_seq.add_module("mfcc_output", nn.LeakyReLU())

    # 6 x 1292 -> 12 x 646 -> 24 x 323 -> 7752
    self.tonnetz_seq = nn.Sequential()
    for index in range(sequence_len):
      self.tonnetz_seq.add_module(f"tonnetz_Conv1d_{index}", nn.Conv1d(tonnetz_channels[0][index], tonnetz_channels[1][index], kernel_size=kernel_sizes[index], stride=strides[index], padding=paddings[index]))
      self.tonnetz_seq.add_module(f"tonnetz_BN1d_{index}", nn.BatchNorm1d(tonnetz_channels[1][index]))
      self.tonnetz_seq.add_module(f"tonnetz_LeakyReLU_{index}", nn.LeakyReLU())
      self.tonnetz_seq.add_module(f"tonnetz_Pool_{index}", nn.MaxPool1d(pool_sizes[index]))
    self.tonnetz_seq.add_module("tonnetz_Flatten", nn.Flatten())
    self.tonnetz_seq.add_module("tonnetz_fc", nn.Linear(7752, 6))
    self.tonnetz_seq.add_module("tonnetz_output", nn.LeakyReLU())

    # 1 x 1292 -> 2 x 646 -> 4 x 323 -> 1292
    self.zcr_seq = nn.Sequential()
    for index in range(sequence_len):
      self.zcr_seq.add_module(f"zcr_Conv1d_{index}", nn.Conv1d(zcr_channels[0][index], zcr_channels[1][index], kernel_size=kernel_sizes[index], stride=strides[index], padding=paddings[index]))
      self.zcr_seq.add_module(f"zcr_BN1d_{index}", nn.BatchNorm1d(zcr_channels[1][index]))
      self.zcr_seq.add_module(f"zcr_LeakyReLU_{index}", nn.LeakyReLU())
      self.zcr_seq.add_module(f"zcr_Pool_{index}", nn.MaxPool1d(pool_sizes[index]))
    self.zcr_seq.add_module("zcr_Flatten", nn.Flatten())
    self.zcr_seq.add_module("zcr_fc", nn.Linear(1292, 1))
    self.zcr_seq.add_module("zcr_output", nn.LeakyReLU())

    self.fc = nn.Linear(30, 1)
  
  def forward(self, x) -> float:
    tempogram = self.tempogram_seq(x[:, 0:10, :])  # 0 ~ 9
    rms = self.rms_seq(x[:, 10:11, :])               # 10
    mfcc = self.mfcc_seq(x[:, 11:23, :])          # 11 ~ 22
    tonnetz = self.tonnetz_seq(x[:, 23:29, :])    # 23 ~ 28
    zcr = self.zcr_seq(x[:, 29:30, :])               # 29
    x = torch.cat((tempogram, rms, mfcc, tonnetz, zcr), dim=1)
    x = self.fc(x)
    return x

def model_select() -> ReprMPL | ReprLinearRegression | SpecCNN | FeatCNN:
  if MODEL_TYPE == MODEL_LR:
    return ReprLinearRegression()
  if MODEL_TYPE == MODEL_SPEC:
    return SpecCNN()
  if MODEL_TYPE == MODEL_FEAT:
    return FeatCNN()
  if True:
    return ReprMPL()