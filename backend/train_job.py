import numpy as np
from train import data_load, valid_index, new_file_path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pydantic import BaseModel
from model import Model

class TrainJob(BaseModel):
  status: bool = False
  now_epoch: int = 0
  train_dataset_dir: str = None
  target_dataset_dir: str = None
  train_model_dir: str = None
    
  def __call__(self) -> None:
    try:
      self.status = True
      self.train()
    except Exception as e:
      print(e)
    finally:
      self.status = False
  
  def get_status(self) -> bool:
    return self.status

  def get_progress(self) -> int:
    return self.now_epoch
  
  def get_model_dir(self) -> str:
    return self.train_model_dir

  def train(self) -> None:
    train_array, target_array = data_load("data/aug_music")

    model = Model()
    learning_rate = 0.002
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 1000
    
    train_loss_history = []
    valid_loss_history = []
    for epoch in range(num_epochs):
      self.now_epoch = epoch
      train_loss_list = np.array([])
      valid_loss_list = np.array([])
      optimizer.zero_grad()
      index_list = valid_index(train_array, 10)
      for index, (train, target) in enumerate(zip(train_array, target_array)):
        train = torch.from_numpy(train).float()
        target = torch.from_numpy(target).float()
        if index in index_list:
          with torch.set_grad_enabled(False):
            model.eval()
            output = model(train)
            valid_loss_list = np.append(valid_loss_list, criterion(output, target).item())
        else:
          with torch.set_grad_enabled(True):
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
    plt.savefig(new_file_path("figure", "loss", ".png"))

    self.train_model_dir = new_file_path("model", "train", ".pth")
    torch.save(model.state_dict(), self.train_model_dir)
