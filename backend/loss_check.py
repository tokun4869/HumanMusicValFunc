import torch
import torch.nn as nn
import numpy as np
import torchaudio
from torch.utils.data import DataLoader
from module.dataset import TrainDataset
from module.io import get_dataset_path, get_file_name_list, load_dataset
from module.loss import HumanMusicValLoss
from module.const import *

class LossCheckModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(60, SAMPLE_RATE*LENGTH)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


if __name__ == "__main__":
    model = LossCheckModel()
    criterion = HumanMusicValLoss(f"{MODEL_ROOT}/#3 - model_ReprMLP{MODEL_EXT}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  
    print("    | dataset")
    dataset_path = get_dataset_path()
    rank_list = [1, 10, 6, 8, 9, 2, 4, 5, 3, 7]
    file_name_list = get_file_name_list(f"{MUSIC_ROOT}/{DATASET_TYPE}/{MODE}_{LISTEN_KEY}")
    input_list, target_list = load_dataset(dataset_path, rank_list, file_name_list)
    dataset = TrainDataset(input_list, target_list)
    train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=True)

    print("#02 | === TRAIN ===")
    for epoch in range(NUM_EPOCHS):
        train_loss_list = []
        data_loss_list = []
        optimizer.zero_grad()
        for feature, _ in train_loader:
            feature = feature.to(torch.float).unsqueeze(dim=1) if MODEL_TYPE == MODEL_SPEC else feature.to(torch.float)
            model.train()
            output = model(feature)
            loss = criterion(output)
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.view(-1).tolist())
            data_loss_list.append(criterion.feature_forward(feature).view(-1).tolist())

        print(f"    | Epoch:[ {epoch+1:>3} / {NUM_EPOCHS:>3} ], TrainLoss:[ {np.mean(train_loss_list):7.2f} ], DataLoss:[ {np.mean(data_loss_list):7.2f} ]")
        torchaudio.save(uri=f"{MUSIC_ROOT}/loss_check{MUSIC_EXT}", src=output.detach(), sample_rate=SAMPLE_RATE, format="mp3")

