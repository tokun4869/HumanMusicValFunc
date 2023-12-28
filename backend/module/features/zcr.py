import torch

def calc_zcr(y: torch.Tensor, frame_length: int=2048, hop_length: int=512, device: torch.device=torch.device("cpu")) -> torch.Tensor:
    n_wins = y.shape[-1] // hop_length if y.shape[-1] % hop_length == 0 else y.shape[-1] // hop_length + 1
    zc_rate = torch.zeros(y.shape[0], n_wins).to(device)
    zc = torch.empty(y.shape).to(device)
    zc[:, 0] = False
    zc[:, 1:] = torch.where(torch.sign(y[:, 1:]) * torch.sign(y[:, :-1]) == -1, True, False)

    for win_index in range(n_wins):
        start = max(win_index * hop_length, 0)
        end = min(start + frame_length, y.shape[-1])
        zc_count = (zc[:, start:end] == True).sum(dim=-1)
        zc_rate[:, win_index] = zc_count / (end-start)
    zc_rate = zc_rate.unsqueeze(1)

    return zc_rate