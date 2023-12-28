import torch
import torchaudio
from module.const import *


def novelty_function(y: torch.Tensor, n_fft: int=N_FFT, hop_length: int=HOP_LENGTH, device: torch.device=torch.device("cpu")) -> torch.Tensor:
    transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
    transform = transform.to(device)
    raw_s: torch.Tensor = transform(y)
    S = torch.log(torch.abs(raw_s.real)+1e-9)
    batch_size, _, n_timesteps = S.shape
    spectral_novelty = torch.zeros(batch_size, n_timesteps).to(device)
    tmp = S[:, :, 1:] - S[:, :, :-1]
    tmp = torch.where(tmp < 0, 0, tmp)
    spectral_novelty[:, 0:-1] = torch.sum(tmp, dim=-2)
    spectral_novelty /= torch.max(spectral_novelty)

    return spectral_novelty


def calc_tempogram(y: torch.Tensor, frame_length: int=10, hop_length: int=1, device: torch.device=torch.device("cpu")) -> torch.Tensor:
    nf = novelty_function(y, device=device)
    n_wins = nf.shape[-1] // hop_length if nf.shape[-1] % hop_length == 0 else nf.shape[-1] // hop_length + 1
    odf_frame = torch.zeros(nf.shape[0], n_wins, frame_length).to(device)

    nf_shape_end = nf.shape[-1]
    padding = [0] * len(nf.shape) * 2
    padding[-1] = frame_length // 2
    padding[-2] = frame_length // 2
    padding.reverse()
    padding = tuple(padding)
    nf = torch.nn.functional.pad(nf, pad=padding)

    window = torch.hann_window(frame_length).to(device)
    window = torch.unsqueeze(window, dim=0)
    window = window.expand(nf.shape[0], window.shape[-1])

    for win_index in range(n_wins):
        start = max(win_index * hop_length, 0)
        end = min(start + frame_length, nf.shape[-1])
        odf_frame[:, win_index] = nf[:, start:end] * window
    
    odf_frame = odf_frame[:, :nf_shape_end]
    n_pad = 2*odf_frame.shape[-2]-1

    S = torch.abs(torch.fft.rfft(odf_frame, n=n_pad, dim=-2)) ** 2
    autocorr = torch.fft.irfft(S, n=n_pad, dim=-2)
    
    subslice = [slice(None)] * autocorr.ndim
    subslice[-2] = slice(odf_frame.shape[-2])
    autocorr_slice: torch.Tensor = autocorr[tuple(subslice)]

    autocorr_slice = torch.transpose(autocorr_slice, dim0=-1, dim1=-2)

    return autocorr_slice