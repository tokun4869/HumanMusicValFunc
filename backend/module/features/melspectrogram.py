import torch
import torchaudio.transforms as t
from module.const import *

def calc_melspectrogram(y: torch.Tensor) -> torch.Tensor:
  amp_spec = t.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)(y)
  db_spec = t.AmplitudeToDB()(amp_spec)
  return db_spec