import numpy as np
import torch
import torch.nn as nn
import csv
import matplotlib.pyplot as plt

from pydantic import BaseModel
from model import Model
from io_module import get_new_file_path, get_file_list
from feature_module import music2feature
from transform_module import data_augmentation
from static_value import *

class TrainJob(BaseModel):
  status: str = STATUS_BEFORE
  file_list: list = []
  rank_list: "list[int]" = []
  now_epoch: int = 0
  num_epochs: int = 100
  train_model_dir: str = None
  error: str = None

  def initialize(self) -> None:
    self.status = STATUS_BEFORE
    self.rank_list = []
    self.now_epoch = 0
    self.train_model_dir = None
    self.error = None
    
  def __call__(self, file_name_list: str, rank_list: str) -> None:
    try:
      self.set_file_list(file_name_list)
      self.rank_list = rank_list
      self.status = STATUS_INPROGRESS
      self.train(rank_list)
    except Exception as e:
      self.status = STATUS_ERROR
      self.error = str(e)
    
  def set_file_list(self, file_name_list: "list[str]") -> None:
    self.file_list = get_file_list(file_name_list)
    
  def get_rank_list(self) -> "list[int]":
    return self.rank_list
  
  def get_status(self) -> bool:
    return self.status

  def get_now_epoch(self) -> int:
    return self.now_epoch
  
  def get_num_epochs(self) -> int:
    return self.num_epochs
  
  def get_model_dir(self) -> str:
    return self.train_model_dir
  
  def get_error(self) -> str:
    return self.error
  
  def get_feature(self, file: "np.ndarray[np.float32]") -> "np.ndarray[np.float32]":
    return music2feature(data_augmentation(file))
  
  def get_value(self, rank: int) -> float:
    base_value = 10 - rank
    alpha = 10.0
    return base_value * alpha

  def train(self, rank_list: "list[int]") -> None:
    # モデル設定
    model = Model()
    learning_rate = 0.001
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss_history = []

    # 学習
    for epoch in range(self.num_epochs):
      self.now_epoch = epoch
      train_loss_list = np.array([])
      feature_list = np.array([])
      optimizer.zero_grad()
      for file, rank in zip(self.file_list, rank_list):
        feature = self.get_feature(file)
        value = self.get_value(rank)
        feature_list = feature if feature_list.size == 0 else np.vstack([feature_list, feature])
        input = torch.from_numpy(feature).float()
        target = torch.tensor([value])
        model.train()
        output = model(input)
        loss = criterion(output, target)
        train_loss_list = np.append(train_loss_list, loss.item())
      loss.backward()
      optimizer.step()
      train_loss_history.append(train_loss_list.mean())
    
    with open(get_new_file_path(FEATURE_ROOT, "feature", ".csv"), "w") as f:
      writer = csv.writer(f)
      writer.writerows(feature_list)
    
    plt.plot(train_loss_history)
    plt.savefig(get_new_file_path(GRAPH_ROOT, "loss", ".png"))
    plt.clf()

    self.train_model_dir = get_new_file_path(MODEL_ROOT, "model", ".pth")
    torch.save(model.state_dict(), self.train_model_dir)
    self.status = STATUS_FINISH
