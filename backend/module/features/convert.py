import torch
import module.features as f

# kernel, fft_win_length, fft_hop_length = f.kernel_matrix()

def music2feature(y: torch.Tensor) -> torch.Tensor:
  feature_list = []
  feature_list.append(f.calc_tempogram(y))
  feature_list.append(f.calc_rms(y))
  feature_list.append(f.calc_mfcc(y))
  feature_list.append(f.calc_centroid(y))
  # feature_list.append(f.calc_tonnetz(y, kernel=kernel, fft_win_length=fft_win_length, fft_hop_length=fft_hop_length))
  feature_list.append(f.calc_zcr(y))
  feature_tensor = torch.cat(feature_list, dim=1)
  return feature_tensor

def feature2represent(feature_tensor: torch.Tensor) -> torch.Tensor:
  feature_mean_tensor = f.feature_mean(feature_tensor)
  feature_var_tensor = f.feature_var(feature_tensor)
  represent_tensor = torch.cat([feature_mean_tensor, feature_var_tensor], dim=1)
  return represent_tensor

def music2represent(y: torch.Tensor) -> torch.Tensor:
  feature_tensor = music2feature(y)
  represent_tensor = feature2represent(feature_tensor)
  return represent_tensor

def music2melspectrogram(y: torch.Tensor) -> torch.Tensor:
  return f.calc_melspectrogram(y)