import numpy as np
import random
import glob
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import Model

def data_load(dir: str) -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  train = np.load(dir + "/train.npy")
  target = np.load(dir + "/target.npy")
  return train, target

def valid_index(array: "np.ndarray[np.float32]", n_index: int):
  index_list = []
  while len(index_list) < n_index:
    index = random.randint(0, array.shape[0])
    if index not in index_list:
      index_list.append(index)
  return index_list

def new_file_path(dir: str, base: str, ext: str):
  file_name_list = glob.glob("{}/{}_*{}".format(dir, base, ext))
  max_index = -1
  for file_name in file_name_list:
    index = int(file_name[len(dir)+len("/")+len(base)+len("_") : -len(ext)])
    if max_index < index:
      max_index = index
  path = "{}/{}_{}{}".format(dir, base, max_index+1, ext)
  return path

def train():
  train_array, target_array = data_load("data/aug_music")

  model = Model()
  learning_rate = 0.002
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  num_epochs = 1000

  print("=== Train Start! ===")
  train_loss_history = []
  valid_loss_history = []
  for epoch in range(num_epochs):
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
          # print("[Eval] output: {}, target: {}".format(output, target))
      else:
        with torch.set_grad_enabled(True):
          model.train()
          output = model(train)
          loss = criterion(output, target)
          train_loss_list = np.append(train_loss_list, loss.item())
          # print("[Train] output: {}, target: {}".format(output, target))
    loss.backward()
    optimizer.step()
    train_loss_history.append(train_loss_list.mean())
    valid_loss_history.append(valid_loss_list.mean())
    if (epoch + 1) % 10 == 0:
      print("Epoch [{:3d}/{:3d}]: train_loss = {:.4f}, valid_loss = {:.4f}".format(epoch, num_epochs, train_loss_list.mean(), valid_loss_list.mean()))
  
  plt.plot(train_loss_history, label="train")
  plt.plot(valid_loss_history, label="valid", alpha=0.5)
  plt.legend()
  plt.savefig(new_file_path("figure", "loss", ".png"))

  path = new_file_path("model", "train", ".pth")
  torch.save(model.state_dict(), path)
  print("Model is Saved at {}".format(path))
  return path

if __name__ == "__main__":
  path = train()