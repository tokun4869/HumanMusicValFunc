import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pydantic import BaseModel
from dataset import Dataset
from model import Model, SpecCNNModel
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from io_module import get_new_file_path, load_dataset
from feature_module import get_feature_name_list
from static_value import *

class TrainJob(BaseModel):
  status: str = STATUS_BEFORE
  now_epoch: int = 0
  train_model_dir: str = None
  error: str = None

  def initialize(self) -> None:
    self.status = STATUS_BEFORE
    self.now_epoch = 0
    self.train_model_dir = None
    self.error = None
    
  def __call__(self, rank_list: str) -> None:
    try:
      self.status = STATUS_INPROGRESS
      dataset_base = FEAT_DATASET_BASE if IS_FEAT else SPEC_DATASET_BASE
      feature_list, target_list = load_dataset(DATASET_ROOT + dataset_base, rank_list)
      dataset = Dataset(feature_list, target_list)
      self.train(dataset)
    except Exception as e:
      self.status = STATUS_ERROR
      self.error = str(e)
  
  def get_status(self) -> bool:
    return self.status

  def get_now_epoch(self) -> int:
    return NUM_EPOCHS
  
  def get_num_epochs(self) -> int:
    return self.num_epochs
  
  def get_model_dir(self) -> str:
    return self.train_model_dir

  def get_error(self) -> str:
    return self.error
  
  def get_value(self, rank: int) -> float:
    base_value = 10 - rank
    alpha = 10.0
    return base_value * alpha

  def train(self, dataset: Dataset) -> None:
    # モデル設定
    model = Model() if IS_FEAT else SpecCNNModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loss_history = []
    valid_loss_history = []
    
    kf = KFold(n_splits=int(len(dataset)/2))

    # 学習
    for epoch in range(NUM_EPOCHS):
      self.now_epoch = epoch
      train_loss_list = np.array([])
      valid_loss_list = np.array([])
      optimizer.zero_grad()
      
      for train_index, valid_index in kf.split(range(len(dataset))):
        train_dataset = Subset(dataset, train_index)
        valid_dataset = Subset(dataset, valid_index)

        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, BATCH_SIZE, shuffle=False)

        for feature, rank in train_loader:
          feature = feature.to(torch.float).view(-1, len(get_feature_name_list()))
          value = self.get_value(rank).to(torch.float).view(-1, 1)
          model.train()
          output = model(feature)
          loss = criterion(output, value)
          train_loss_list = np.append(train_loss_list, loss.item())
          loss.backward()
          optimizer.step()
        
        for feature, rank in valid_loader:
          with torch.no_grad():
            feature = feature.to(torch.float).view(-1, len(get_feature_name_list()))
            value = self.get_value(rank).to(torch.float).view(-1, 1)
            model.eval()
            output = model(feature)
            loss = criterion(output, value)
            valid_loss_list = np.append(valid_loss_list, loss.item())
        
      train_loss_history.append(train_loss_list.mean())
      valid_loss_history.append(valid_loss_list.mean())
    
    plt.plot(train_loss_history, label="train")
    plt.plot(valid_loss_history, label="valid", alpha=0.5)
    plt.legend()
    plt.savefig(get_new_file_path(GRAPH_ROOT, "loss", ".png"))
    plt.clf()

    self.train_model_dir = get_new_file_path(MODEL_ROOT, "model", ".pth")
    torch.save(model.state_dict(), self.train_model_dir)
    self.status = STATUS_FINISH
