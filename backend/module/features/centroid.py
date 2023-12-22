import torch
import torchaudio.transforms as t
from module.const import *

def calc_centroid(y: torch.Tensor) -> torch.Tensor:
    return t.SpectralCentroid(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)(y).unsqueeze(1)