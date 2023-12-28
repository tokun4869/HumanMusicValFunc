import torch
from module.const import *

def calc_rms(y: torch.Tensor, frame_length: int=2048, hop_length: int=HOP_LENGTH, device: torch.device=torch.device("cpu")) -> torch.Tensor:
    n_wins = y.shape[-1] // hop_length if y.shape[-1] % hop_length == 0 else y.shape[-1] // hop_length + 1
    rms_value = torch.zeros(y.shape[0], n_wins).to(device)
    y_square = y ** 2

    for win_index in range(n_wins):
        start = max(win_index * hop_length, 0)
        end = min(start + frame_length, y.shape[-1])
        rms_value[:, win_index] = torch.sqrt(torch.mean(y_square[:, start:end], dim=-1))
    rms_value = rms_value.unsqueeze(1)

    return rms_value