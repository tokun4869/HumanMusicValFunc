import torch.nn as nn
import torch.nn.functional as F

from static_value import SAMPLE_RATE

class Model(nn.Module):
  def __init__(self) -> None:
    super(Model, self).__init__()
    self.fc1 = nn.Linear(41, 10)
    self.fc2 = nn.Linear(10, 1)
  
  def forward(self, x) -> float:
    x = F.leaky_relu(self.fc1(x))
    x = self.fc2(x)
    return x


class LinearRegresssion(nn.Module):
  def __init__(self) -> None:
    super(LinearRegresssion, self).__init__()
    self.linear = nn.Linear(41, 1)
  
  def forward(self, x) -> float:
    return self.linear(x)


class SpecCNNModel(nn.Module):
  def __init__(self) -> None:
    super(SpecCNNModel, self).__init__()
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
      self.sequence.add_module("Conv2d_{}".format(index), nn.Conv2d(in_channels[index], out_channels[index], kernel_size=kernel_sizes[index], stride=strides[index], padding=paddings[index]))
      self.sequence.add_module("BN2d_{}".format(index), nn.BatchNorm2d(out_channels[index]))
      self.sequence.add_module("LeakyReLU_{}".format(index), nn.LeakyReLU())
      self.sequence.add_module("Pool_{}".format(index), nn.MaxPool2d(pool_sizes[index]))
    self.sequence.add_module("Flatten", nn.Flatten())
    self.sequence.add_module("fc", nn.Linear(20608, 1))

  def forward(self, x) -> float:
    x = self.sequence(x)
    return x