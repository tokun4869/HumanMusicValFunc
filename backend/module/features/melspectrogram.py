import torch
import torchaudio.transforms as t
from module.const import *

def calc_melspectrogram(y: torch.Tensor, device: torch.device=torch.device("cpu")) -> torch.Tensor:
  transform = torch.nn.Sequential()
  transform.add_module("melspectrogram", t.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH))
  transform.add_module("amptodb", t.AmplitudeToDB())
  transform = transform.to(device)
  return transform(y)