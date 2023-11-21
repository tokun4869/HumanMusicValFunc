import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from model import Model, SpecCNNModel
from io_module import get_file_name_list, load_dataset, get_new_file_path
from dataset import Dataset
from feature_module import get_feature_name_list
from static_value import *

def get_value(rank: torch.Tensor) -> torch.Tensor:
  base_value = 10 - rank
  alpha = 10.0
  return base_value * alpha

if __name__ == "__main__":
  dataset_dir_list = ["data/list/train_feat_dataset_0.csv"] if IS_FEAT else ["data/list/train_spec_dataset_0.csv"]
  answer_list_list = [[1, 10, 6, 8, 9, 2, 4, 5, 3, 7]]
  file_name_list_list = [get_file_name_list(MUSIC_ROOT+TRAIN_LISTEN_DIR)]
  label_list = ["CNN"]

  for dataset_dir, answer_list, file_name_list, label in zip(dataset_dir_list, answer_list_list, file_name_list_list, label_list):
    print("#01 | === SETUP ===")
    print("    | model")
    model = Model() if IS_FEAT else SpecCNNModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("    | dataset")
    input_list, target_list = load_dataset(dataset_dir, answer_list, file_name_list)
    dataset = Dataset(input_list, target_list)
    kf = KFold(n_splits=int(len(dataset)/2))

    print("    | loss_history")
    train_loss_history = []
    valid_loss_history = []

    print("#02 | === TRAIN ===")
    for epoch in range(NUM_EPOCHS):
      # print("    | Epoch:[{:>3}/{:>3}]".format(epoch, NUM_EPOCHS))
      train_loss_list = np.array([])
      valid_loss_list = np.array([])
      optimizer.zero_grad()

      for train_index, valid_index in kf.split(range(len(dataset))):
        # print("    | train idx : {}".format(train_index))
        # print("    | valid idx : {}".format(valid_index))
        # print("    | setup dataloader")
        train_dataset = Subset(dataset, train_index)
        valid_dataset = Subset(dataset, valid_index)
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, BATCH_SIZE, shuffle=False)

        # print("    | setup loss_list")
        train_loss_list = []
        valid_loss_list = []

        # print("    | train")
        for feature, rank in train_loader:
          feature = feature.to(torch.float) if IS_FEAT else feature.to(torch.float).view(feature.size()[0], 1,  feature.size()[1], feature.size()[2])
          value = get_value(rank).to(torch.float).view(-1, 1)
          model.train()
          output = model(feature)
          loss = criterion(output, value)
          train_loss_list = np.append(train_loss_list, loss.item())
          loss.backward()
          optimizer.step()
        
        # print("    | valid")
        for feature, rank in valid_loader:
          with torch.no_grad():
            feature = feature.to(torch.float) if IS_FEAT else feature.to(torch.float).view(feature.size()[0], 1, feature.size()[1], feature.size()[2])
            value = get_value(rank).to(torch.float).view(-1, 1)
            model.eval()
            output = model(feature)
            loss = criterion(output, value)
            valid_loss_list = np.append(valid_loss_list, loss.item())
        
      # print("    | set loss")
      train_loss_history.append(train_loss_list.mean())
      valid_loss_history.append(valid_loss_list.mean())
      print("    | Epoch:[{:>3}/{:>3}], TrainLoss:[{:.2f}], ValidLoss:[{:.2f}]".format(epoch, NUM_EPOCHS, train_loss_list.mean(), valid_loss_list.mean()))
    
    print("#03 | === DRAW LOSS ===")
    plt.plot(train_loss_history, label="train")
    plt.plot(valid_loss_history, label="valid", alpha=0.5)
    plt.legend()
    plt.savefig(get_new_file_path(GRAPH_ROOT, "loss", ".png"))
    plt.clf()
    
    print("#04 | === SAVE MODEL ===")
    train_model_dir = get_new_file_path(MODEL_ROOT, "model", ".pth")
    torch.save(model.state_dict(), train_model_dir)