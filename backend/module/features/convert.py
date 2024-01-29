import torch
from module.const import EXTRACTOR_FEAT, EXTRACTOR_REPR, EXTRACTOR_SPEC
import module.features as f

# kernel, fft_win_length, fft_hop_length = f.kernel_matrix()

def music2feature(y: torch.Tensor, device: torch.device=torch.device("cpu")) -> torch.Tensor:
  feature_list = []
  feature_list.append(f.calc_tempogram(y, device=device))
  feature_list.append(f.calc_rms(y, device=device))
  feature_list.append(f.calc_mfcc(y, device=device))
  feature_list.append(f.calc_centroid(y, device=device))
  feature_list.append(f.calc_zcr(y, device=device))
  feature_tensor = torch.cat(feature_list, dim=1)
  return feature_tensor

def feature2represent(feature_tensor: torch.Tensor) -> torch.Tensor:
  feature_mean_tensor = f.feature_mean(feature_tensor)
  feature_var_tensor = f.feature_var(feature_tensor)
  represent_tensor = torch.cat([feature_mean_tensor, feature_var_tensor], dim=1)
  return represent_tensor

def music2represent(y: torch.Tensor, device: torch.device=torch.device("cpu")) -> torch.Tensor:
  feature_tensor = music2feature(y, device)
  represent_tensor = feature2represent(feature_tensor)
  return represent_tensor

def music2melspectrogram(y: torch.Tensor, device: torch.device=torch.device("cpu")) -> torch.Tensor:
  return f.calc_melspectrogram(y, device)

def music2input(y: torch.Tensor, extractor: str, device: torch.device=torch.device("cpu")) -> torch.Tensor:
  if extractor == EXTRACTOR_REPR:
    return f.music2represent(y, device)
  if extractor == EXTRACTOR_SPEC:
    return f.music2melspectrogram(y, device)
  if extractor == EXTRACTOR_FEAT:
    return f.music2feature(y, device)