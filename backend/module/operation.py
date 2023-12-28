import time
import torch
import torchinfo
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from module.model import Model
from module.dataset import TrainDataset, TestDataset
from module.io import load_dataset, load_model, get_new_file_path
from module.feature import musics2input
from module.const import *
from module.util import EarlyStop


def train_operation(dataset_path: str, rank_list: list[int], file_name_list: list[str], label_name: str=None, extractor: str = EXTRACTOR_TYPE, head: str = HEAD_TYPE, set_now_epoch = None, set_status = None, set_model_path = None):
  print("#01 | === SETUP ===")
  print("    | model")
  torch.manual_seed(SEED)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = Model(extractor, head, device)
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  file_base = "model" if label_name == None else f"model_{label_name}"
  model_path = get_new_file_path(MODEL_ROOT, file_base, MODEL_EXT)
  early_stop = EarlyStop(model_path=model_path)
  
  print("    | dataset")
  input_list, target_list = load_dataset(dataset_path, rank_list, file_name_list)
  dataset = TrainDataset(input_list, target_list)
  kf = KFold(n_splits=int(len(dataset)/BATCH_SIZE))

  print("    | loss_history")
  train_loss_history = []
  valid_loss_history = []

  print("    | time_history")
  time_history = []

  print("#02 | === TRAIN ===")
  for epoch in range(NUM_EPOCHS):
    start_time = time.time()

    if set_now_epoch != None: set_now_epoch(epoch)
    train_loss_list = np.array([])
    valid_loss_list = np.array([])
    optimizer.zero_grad()

    for train_index, valid_index in kf.split(range(len(dataset))):
      train_dataset = Subset(dataset, train_index)
      valid_dataset = Subset(dataset, valid_index)
      train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
      valid_loader = DataLoader(valid_dataset, BATCH_SIZE, shuffle=False)

      train_loss_list = []
      valid_loss_list = []

      for feature, value in train_loader:
        value = value.to(torch.float).view(-1, 1)
        feature = feature.to(device)
        value = value.to(device)
        model.train()
        output = model(feature)
        loss = criterion(output, value)
        train_loss_list = np.append(train_loss_list, loss.item())
        loss.backward()
        optimizer.step()
      
      for feature, value in valid_loader:
        with torch.no_grad():
          value = value.to(torch.float).view(-1, 1)
          feature = feature.to(device)
          value = value.to(device)
          model.eval()
          output = model(feature)
          loss = criterion(output, value)
          valid_loss_list = np.append(valid_loss_list, loss.item())
      
    train_loss_history.append(train_loss_list.mean())
    valid_loss_history.append(valid_loss_list.mean())

    end_time = time.time()
    epoch_time = end_time - start_time
    time_history.append(epoch_time)

    print(f"    | Epoch:[ {epoch+1:>3} / {NUM_EPOCHS:>3} ], TrainLoss:[ {train_loss_list.mean():7.2f} ], ValidLoss:[ {valid_loss_list.mean():7.2f} ], Time:[ {epoch_time:7.2f} ]")
    if early_stop(loss=valid_loss_list.mean(), model=model):
      break

  print("#03 | === SAVE LOSS HISTORY ===")
  file_base = "loss" if label_name == None else f"loss_{label_name}"
  with open(get_new_file_path(LOSS_ROOT, file_base, LIST_EXT), "w") as f:
    writer = csv.writer(f)
    writer.writerow(train_loss_history)
    writer.writerow(valid_loss_history)

  print("#04 | === DRAW LOSS ===")
  file_base = "graph" if label_name == None else f"graph_{label_name}"
  plt.plot(train_loss_history, label="train")
  plt.plot(valid_loss_history, label="valid", alpha=0.5)
  plt.legend()
  plt.savefig(get_new_file_path(GRAPH_ROOT, file_base, GRAPH_EXT))
  plt.clf()
  
  print("#05 | === SAVE MODEL ===")
  # file_base = "model" if label_name == None else f"model_{label_name}"
  # model_path = get_new_file_path(MODEL_ROOT, file_base, MODEL_EXT)
  torch.save(model.state_dict(), model_path)
  if set_model_path != None: set_model_path(model_path)
  if set_status != None: set_status(STATUS_FINISH)

  print("#06 | === SAVE TIME HISTORY ===")
  file_base = "time" if label_name == None else f"time_{label_name}"
  with open(get_new_file_path(TIME_ROOT, file_base, LIST_EXT), "w") as f:
    writer = csv.writer(f)
    writer.writerow(time_history)


def test_operation(model_path: str, sound_list: list, target_list: list[int], label_name: str=None) -> "list[float]":
  print("#01 | === SETUP ===")
  print("    | model")
  torch.manual_seed(SEED)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = load_model(model_path, device=device)

  print("    | dataset")
  input_list = musics2input(sound_list)
  test_dataset = TestDataset(input_list)
  test_loader = DataLoader(test_dataset, len(test_dataset), shuffle=False)

  print("    | time_history")
  time_history = []

  print("#02 | === TEST ===")
  start_time = time.time()

  for feature in test_loader:
    feature = feature.to(torch.float).to(device)
    output = model(feature)
    output_list = torch.squeeze(output).tolist()
  
  end_time = time.time()
  epoch_time = end_time - start_time
  time_history.append(epoch_time)

  print("#03 | === SAVE RESULT ===")
  file_base = "result" if label_name == None else f"result_{label_name}"
  with open(get_new_file_path(RESULT_ROOT, file_base, LIST_EXT), "w") as f:
    writer = csv.writer(f)
    writer.writerow(output_list)
  
  print("#04 | === SAVE TIME ===")
  file_base = "time_test" if label_name == None else f"time_test_{label_name}"
  with open(get_new_file_path(TIME_ROOT, file_base, LIST_EXT), "w") as f:
    writer = csv.writer(f)
    writer.writerow(time_history)
    writer.writerow(target_list)
  
  return output_list