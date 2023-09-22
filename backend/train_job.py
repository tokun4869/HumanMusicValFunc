import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pydantic import BaseModel
from model import Model
from io_module import get_new_file_path
from static_value import *

class TrainJob(BaseModel):
  status: str = STATUS_BEFORE
  now_epoch: int = 0
  train_model_dir: str = None
  error: str = None
    
  def __call__(self, train_data_dir: str, target_data_dir: str) -> None:
    try:
      self.status = STATUS_INPROGRESS
      self.train(train_data_dir, target_data_dir)
    except Exception as e:
      self.error = str(e)
  
  def get_status(self) -> bool:
    return self.status

  def get_progress(self) -> int:
    return self.now_epoch
  
  def get_model_dir(self) -> str:
    return self.train_model_dir
  
  def get_error(self) -> str:
    return self.error
  
  def get_dataset(self, train_data_dir: str, target_data_dir: str) -> "tuple[np.ndarray[np.float32], np.ndarray[np.float32]]":
    train = np.load(train_data_dir)
    target = np.load(target_data_dir)
    return train, target

  def get_valid_index_list(self, array: "np.ndarray[np.float32]", n_index: int):
    index_list = []
    while len(index_list) < n_index:
      index = random.randint(0, array.shape[0])
      if index not in index_list:
        index_list.append(index)
    return index_list

  def train(self, train_data_dir: str, target_data_dir: str) -> None:
    train_array, target_array = self.get_dataset(train_data_dir, target_data_dir)

    model = Model()
    learning_rate = 0.002
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 500
    
    train_loss_history = []
    valid_loss_history = []
    for epoch in range(num_epochs):
      self.now_epoch = epoch
      train_loss_list = np.array([])
      valid_loss_list = np.array([])
      optimizer.zero_grad()
      index_list = self.get_valid_index_list(train_array, 10)
      for index, (train, target) in enumerate(zip(train_array, target_array)):
        train = torch.from_numpy(train).float()
        target = torch.from_numpy(target).float()
        if index in index_list:
          with torch.no_grad():
            model.eval()
            output = model(train)
            valid_loss_list = np.append(valid_loss_list, criterion(output, target).item())
        else:
          model.train()
          output = model(train)
          loss = criterion(output, target)
          train_loss_list = np.append(train_loss_list, loss.item())
      loss.backward()
      optimizer.step()
      train_loss_history.append(train_loss_list.mean())
      valid_loss_history.append(valid_loss_list.mean())
    
    plt.plot(train_loss_history, label="train")
    plt.plot(valid_loss_history, label="valid", alpha=0.5)
    plt.legend()
    plt.savefig(get_new_file_path(GRAPH_ROOT, "loss", ".png"))

    self.train_model_dir = get_new_file_path(MODEL_ROOT, "train", ".pth")
    torch.save(model.state_dict(), self.train_model_dir)
    self.status = STATUS_FINISH
