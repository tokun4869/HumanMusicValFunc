import torch
import torchaudio.transforms as t
from module.const import *

def calc_mfcc(y: torch.Tensor, n_mfcc=12, device: torch.device=torch.device("cpu")) -> torch.Tensor:
    transform = torch.nn.Sequential()
    transform.add_module("mfcc", t.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=n_mfcc, melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH}))
    transform = transform.to(device)

    return transform(y)