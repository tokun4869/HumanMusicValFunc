import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from module.model import Model
from module.dataset import TrainDataset, TestDataset
from module.io import get_file_name_list, load_dataset, load_model, get_new_file_path, load_sound_list
from module.const import *
from module.transform import data_augmentation
from module.util import EarlyStop


def dataset_operation(dataset_type: str, mode: str) -> None:
  print(f"#01 | {dataset_type} {mode}")
  input_dir = f"{MUSIC_ROOT}/{dataset_type}/{mode}_{INPUT_KEY}"
  listen_dir = f"{MUSIC_ROOT}/{dataset_type}/{mode}_{LISTEN_KEY}"
  file_name_list = get_file_name_list(listen_dir)
  file_list = load_sound_list(file_name_list)
  num_aug = 3
  file_ext = ".npy"
  list_ext = ".csv"
  
  wave_file_name_list = []
  from_file_name_list = []
  
  for file_name, file in zip(file_name_list, file_list):
    for index in range(num_aug):
      print("    | {} [{}/{}]".format(file_name, index+1, num_aug))
      from_file_name_list.append(file_name)
      new_file = data_augmentation(file)
      file_base = file_name[len(f"{listen_dir}/"):-len(file_ext)]

      wave_file_base = file_base + "_wave"
      new_wave_file_name = get_new_file_path(input_dir, wave_file_base, file_ext)
      print("    | -> {}".format(new_wave_file_name))
      np.save(new_wave_file_name, new_file)
      wave_file_name_list.append(new_wave_file_name)
  
  dataset_dir = f"{DATASET_ROOT}/{dataset_type}/{mode}"
  dataset_base = "wave_dataset"
  with open(get_new_file_path(dataset_dir, dataset_base, list_ext), "w") as f:
    writer = csv.writer(f)
    writer.writerows(zip(wave_file_name_list, from_file_name_list))


def train_operation(dataset_path: str, rank_list: list[int], file_name_list: list[str], label_name: str=None, extractor: str = EXTRACTOR_TYPE, head: str = HEAD_TYPE, set_now_epoch = None, set_status = None, set_model_path = None):
  print("#01 | === SETUP ===")
  print("    | model")
  torch.manual_seed(SEED)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = Model(extractor, head, device, feat_extract=False)
  criterion = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  file_base = "model" if label_name == None else f"model_{label_name}"
  model_path = get_new_file_path(MODEL_ROOT, file_base, MODEL_EXT)
  early_stop = EarlyStop(model_path=model_path)
  
  print("    | dataset")
  input_list, target_list = load_dataset(dataset_path, rank_list, file_name_list)
  data_load_time_start = time.time()
  dataset = TrainDataset(input_list, target_list, extractor=extractor, device=device)
  data_load_time_end = time.time()
  data_load_time = data_load_time_end - data_load_time_start
  kf = KFold(n_splits=10)

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

      for feature, value in train_loader:
        value = value.to(torch.float).view(-1, 1)
        feature = feature.to(device)
        value = value.to(device)
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
  
  print("#03 | === SAVE MODEL ===")
  torch.save(model.state_dict(), model_path)
  if set_model_path != None: set_model_path(model_path)
  if set_status != None: set_status(STATUS_FINISH)

  return train_loss_history, valid_loss_history, time_history, data_load_time


def test_operation(model_path: str, sound_list: list, label: str=None, extractor: str = EXTRACTOR_TYPE) -> "list[float]":
  print("#01 | === SETUP ===")
  print("    | model")
  torch.manual_seed(SEED)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = load_model(model_path, extractor, device=device)

  print("    | dataset")
  input_list = [data_augmentation(y) for y in sound_list]
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

  return output_list, time_history


def retrain_operation(dataset_path: str, rank_list: list[int], file_name_list: list[str], model_path: str, label_name: str=None, extractor: str = EXTRACTOR_TYPE, head: str = HEAD_TYPE, set_now_epoch = None, set_status = None, set_model_path = None):
  print("#01 | === SETUP ===")
  print("    | model")
  torch.manual_seed(SEED)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = load_model(model_path, extractor=extractor, head=head, device=device, feat_extract=False)
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  file_base = "model" if label_name == None else f"model_{label_name}"
  model_path = get_new_file_path(MODEL_ROOT, file_base, MODEL_EXT)
  early_stop = EarlyStop(model_path=model_path)
  
  print("    | dataset")
  input_list, target_list = load_dataset(dataset_path, rank_list, file_name_list)
  data_load_time_start = time.time()
  dataset = TrainDataset(input_list, target_list, extractor=extractor, device=device)
  data_load_time_end = time.time()
  data_load_time = data_load_time_end - data_load_time_start
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
  
  print("#04 | === SAVE MODEL ===")
  # file_base = "model" if label_name == None else f"model_{label_name}"
  # model_path = get_new_file_path(MODEL_ROOT, file_base, MODEL_EXT)
  torch.save(model.state_dict(), model_path)
  if set_model_path != None: set_model_path(model_path)
  if set_status != None: set_status(STATUS_FINISH)

  print("#05 | === SAVE TIME HISTORY ===")
  file_base = "time" if label_name == None else f"time_{label_name}"
  with open(get_new_file_path(TIME_ROOT, file_base, LIST_EXT), "w") as f:
    writer = csv.writer(f)
    writer.writerow(time_history)
  
  return train_loss_history, valid_loss_history, time_history, data_load_time