import torch
import torchaudio.transforms as t
from module.const import *

def calc_centroid(y: torch.Tensor, device: torch.device=torch.device("cpu")) -> torch.Tensor:
    transform = torch.nn.Sequential()
    transform.add_module("centroid", t.SpectralCentroid(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH))
    transform = transform.to(device)
    
    return transform(y).unsqueeze(1)