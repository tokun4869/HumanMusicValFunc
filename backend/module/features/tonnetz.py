import math
import torch
from module.const import *


# 対数周波数ビン数を計算
def get_n_freqs(f_min: int=C1_HZ, f_max: int=B7_HZ, n_oct_bins: int=12) -> int:
    return int(round(math.log2(float(f_max) / f_min) * n_oct_bins)) + 1

# 各対数周波数ビンに対応する周波数を計算
def get_freqs(n_freqs: int, f_min: int=C1_HZ, n_oct_bins: int=12) -> torch.Tensor:
    return f_min * (2 ** (torch.arange(n_freqs) / float(n_oct_bins)))

def get_constant_Q_param(f_min: int=C1_HZ, f_max: int=B7_HZ, n_oct_bins: int=12) -> tuple[int, torch.Tensor, float]:
    n_freqs = get_n_freqs(f_min, f_max, n_oct_bins) # number of freq bins
    freqs = get_freqs(n_freqs, f_min, n_oct_bins) # freqs [Hz]
    Q = 1.0 / (2 ** (1.0 / n_oct_bins) - 1) # Q value
    return n_freqs, freqs, Q


def constant_Q_transform(y: torch.Tensor, f_min: int=C1_HZ, f_max: int=B7_HZ, n_oct_bins: int=12, hop_length: int=512):
    two_pi_j = 2 * torch.pi * 1j
    n_freqs, freqs, Q = get_constant_Q_param(f_min=f_min, f_max=f_max, n_oct_bins=n_oct_bins)
    if hop_length is None: hop_length = int(round(0.01 * SAMPLE_RATE))
    n_wins = y.shape[-1] // hop_length if y.shape[-1] % hop_length == 0 else y.shape[-1] // hop_length + 1 # number of time frames
    spec = torch.zeros([y.shape[0], n_freqs, n_wins], dtype=torch.complex64) # Constant-Q spectrogram

    # Execution
    for k in range(n_freqs):
        win_length = int(round(Q * SAMPLE_RATE / freqs[k].item()))
        offset = int(win_length / 2)

        # Calculate window function (and weight).
        phase = torch.exp(-two_pi_j * Q * torch.arange(win_length) / win_length)
        weight = phase * torch.hann_window(window_length=win_length)
        weight = weight.unsqueeze(dim=0)
        weight = weight.expand((y.shape[0], win_length))

        # Perform Constant-Q Transform.
        for bin_idx in range(n_wins):
            start = bin_idx * hop_length - offset   # フレーム開始点（x番フレーム*フレーム移動量-窓長の半分　窓の中心を全ての周波数で合わせる）．
            end = start + win_length                # フレーム終了点．
            y_start = min(max(0, start), y.shape[-1])
            y_end = min(max(0, end), y.shape[-1])
            win_start = min(max(0, y_start - start), win_length)  # 最初らへん対策
            win_end = min(max(0, y.shape[-1] - start), win_length)      # 最後らへん対策
            win_slice = weight[:, win_start : win_end]
            sig = y[:, y_start : y_end]
            spec[:, k, bin_idx] = (sig * win_slice).sum(dim=1) / win_length

    return spec


def kernel_matrix(f_min: int=C1_HZ, f_max: int=B7_HZ, n_oct_bins: int=12, spThreshold = 0.0054) -> tuple[torch.Tensor, int, int]:
    two_pi_j = 2 * torch.pi * 1j
    n_freqs, freqs, Q = get_constant_Q_param(f_min=f_min, f_max=f_max, n_oct_bins=n_oct_bins)
    fft_win_length = int(2 ** (math.ceil(math.log2(int(round(Q * SAMPLE_RATE) / freqs[0].item())))))
    fft_hop_length = int(fft_win_length / 2)
    sparseKernel = torch.zeros([n_freqs, fft_win_length], dtype=torch.complex64)
    for k in range(n_freqs):
        tmpKernel = torch.zeros(fft_win_length, dtype=torch.complex64)
        freq = freqs[k]
        # N_k 
        N_k = int(float(SAMPLE_RATE * Q) / freq)
        # FFT窓の中心を解析部分に合わせる．
        startWin = int((fft_win_length - N_k) / 2)
        tmpKernel[startWin : startWin + N_k] = (torch.hann_window(N_k) / N_k) * torch.exp(two_pi_j * Q * torch.arange(N_k) / N_k)
        sparseKernel[k] = torch.fft.fft(tmpKernel)  # FFT (kernel matrix)
    
    sparseKernel[abs(sparseKernel) <= spThreshold] = 0  ### 十分小さい値を０にする
    sparseKernel = sparseKernel.conj() / fft_win_length ### 複素共役にする
    sparseKernel = sparseKernel.to_sparse_csr()

    return sparseKernel, fft_win_length, fft_hop_length


def constant_Q_transform_fft(y: torch.Tensor, kernel: torch.Tensor, fft_win_length: int, fft_hop_length: int, hop_length: int=512) -> torch.Tensor:
    if hop_length is None: hop_length = int(round(0.01 * SAMPLE_RATE))
    n_wins = y.shape[-1] // hop_length if y.shape[-1] % hop_length == 0 else y.shape[-1] // hop_length + 1 # number of time frames

    y_add = torch.zeros(y.shape[0], y.shape[-1] + fft_win_length)
    y_add[:, fft_hop_length: -fft_hop_length] = y

    spec = torch.zeros([y.shape[0], kernel.shape[-2], n_wins], dtype=torch.complex64) # Constant-Q spectrogram

    # Execution
    for win_idx in range(n_wins):
        start = win_idx * hop_length
        end = start + fft_win_length
        y_fft: torch.Tensor = torch.fft.fft(y_add[:, start:end])
        y_fft = y_fft.unsqueeze(-1)
        for batch in range(y_fft.shape[0]):
            tmp_mm: torch.Tensor = torch.sparse.mm(kernel, y_fft[batch])
            spec[batch, :, win_idx] = tmp_mm.squeeze()
    
    return spec


def cqt_to_chroma(cqt_spec: torch.Tensor) -> torch.Tensor:
    n_oct = 12
    chroma = torch.zeros((cqt_spec.shape[0], n_oct, cqt_spec.shape[-1]))
    for m in range(n_oct):
        chroma[:, m, :] = torch.mean(cqt_spec[:, m:cqt_spec.shape[-2]:n_oct, :], dim=-2).real
    
    return chroma


def calc_tonnetz(y: torch.Tensor, kernel: torch.Tensor, fft_win_length: int, fft_hop_length: int) -> torch.Tensor:
    cqt = constant_Q_transform_fft(y, kernel, fft_win_length, fft_hop_length)
    chroma = cqt_to_chroma(cqt)
    norm = torch.abs(chroma).sum(dim=-2, keepdim=True)
    chroma_norm = chroma / norm

    phase_scale = torch.Tensor([7.0/6, 7.0/6, 3.0/2, 3.0/2, 2.0/3, 2.0/3]) * torch.pi
    l = torch.arange(0, 12)
    phase = torch.outer(phase_scale, l)
    phase[::2] -= 0.5
    ton = torch.einsum("pc,...ci->...pi", torch.cos(phase), chroma_norm)
    return ton