import torch
import torchaudio.transforms as t
import module.features as f
from module.const import *

def calc_centroid(y: torch.Tensor, device: torch.device=torch.device("cpu")) -> torch.Tensor:
    transform = torch.nn.Sequential()
    transform.add_module("centroid", t.SpectralCentroid(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH))
    transform = transform.to(device)
    
    return torch.nan_to_num(transform(y).unsqueeze(1))
