import torch
import torch.nn as nn
import torchaudio
from module.loss import HumanMusicValLoss
from module.const import *

class LossCheckModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, SAMPLE_RATE*LENGTH)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.relu(self.fc1(x))
        return self.fc2(h)


if __name__ == "__main__":
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LossCheckModel().to(device)
    criterion = HumanMusicValLoss(f"{MODEL_ROOT}/model_{DATASET_MAESTRO}_{EXTRACTOR_REPR}_M2 高木_0{MODEL_EXT}", extractor=EXTRACTOR_REPR, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        feature = torch.rand(BATCH_SIZE, 100, device=device)
        output = model(feature)
        loss = criterion(output)
        loss.backward()
        optimizer.step()

        print(f"    | Epoch:[ {epoch+1:>3} / {NUM_EPOCHS:>3} ], Loss:[ {loss.item():7.2f} ]")
    torchaudio.save(uri=f"{MUSIC_ROOT}/loss_check{MUSIC_EXT}", src=output.detach(), sample_rate=SAMPLE_RATE, format="mp3")

