# ===== ===== ===== =====
# Music -> Feature Module
# ===== ===== ===== =====

import numpy as np
import librosa

from static_value import SAMPLE_RATE

# テンポ
def bpm(y: "np.ndarray[np.float32]") -> np.float32:
  return librosa.feature.tempo(y=y, sr=SAMPLE_RATE)

# 音圧
def rms(y: "np.ndarray[np.float32]") -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  feature = librosa.feature.rms(y=y)
  feature_mean = feature.mean(axis=1)
  feature_std = np.std(feature, axis=1)
  return feature_mean, feature_std

# 音色
def mfcc(y: "np.ndarray[np.float32]", n_mfcc: int=12, dct_type: int=3) -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  feature = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=n_mfcc, dct_type=dct_type)
  feature_mean = feature.mean(axis=1)
  feature_std = np.std(feature, axis=1)
  return feature_mean, feature_std

# 和音に関する特徴量
def tonnetz(y: "np.ndarray[np.float32]") -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  feature = librosa.feature.tonnetz(y=y, sr=SAMPLE_RATE)
  feature_mean = feature.mean(axis=1)
  feature_std = np.std(feature, axis=1)
  return feature_mean, feature_std

# ゼロ交叉率
def zero_crossing_rate(y: "np.ndarray[np.float32]") -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  feature = librosa.feature.zero_crossing_rate(y=y)
  feature_mean = feature.mean(axis=1)
  feature_std = np.std(feature, axis=1)
  return feature_mean, feature_std

# mel-spectrogram
def melspectrogram(y: "np.ndarray[np.float32]") -> "np.ndarray[[np.float32]]":
  return librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE)

def title2feature(path: str) -> "np.ndarray[np.float32]":
  y, _ = librosa.load(path, sr=SAMPLE_RATE)
  return music2feature(y)

def music2feature(y: "np.ndarray[np.float32]") -> "np.ndarray[np.float32]":
  feature_array = []
  bpm_value = bpm(y)
  rms_mean, rms_std = rms(y)
  mfcc_mean, mfcc_std = mfcc(y)
  tonnetz_mean, tonnetz_std = tonnetz(y)
  zero_crossing_rate_mean, zero_crossing_rate_std = zero_crossing_rate(y)
  feature_array.extend(bpm_value.tolist())
  feature_array.extend(rms_mean.tolist())
  feature_array.extend(rms_std.tolist())
  feature_array.extend(mfcc_mean.tolist())
  feature_array.extend(mfcc_std.tolist())
  feature_array.extend(tonnetz_mean.tolist())
  feature_array.extend(tonnetz_std.tolist())
  feature_array.extend(zero_crossing_rate_mean.tolist())
  feature_array.extend(zero_crossing_rate_std.tolist())
  return np.array(feature_array)

def get_feature_name_list() -> "list[str]":
  feature_name_list = ["bpm_value", "rms_mean", "rms_std", "mfcc_mean_1", "mfcc_mean_2", "mfcc_mean_3", "mfcc_mean_4", "mfcc_mean_5", "mfcc_mean_6", "mfcc_mean_7", "mfcc_mean_8", "mfcc_mean_9", "mfcc_mean_10", "mfcc_mean_11", "mfcc_mean_12", "mfcc_std_1", "mfcc_std_2", "mfcc_std_3", "mfcc_std_4", "mfcc_std_5", "mfcc_std_6", "mfcc_std_7", "mfcc_std_8", "mfcc_std_9", "mfcc_std_10", "mfcc_std_11", "mfcc_std_12", "tonnetz_mean_1", "tonnetz_mean_2", "tonnetz_mean_3", "tonnetz_mean_4", "tonnetz_mean_5", "tonnetz_mean_6", "tonnetz_std_1", "tonnetz_std_2", "tonnetz_std_3", "tonnetz_std_4", "tonnetz_std_5", "tonnetz_std_6", "zcr_mean", "zcr_std"]
  return feature_name_list

def normalization(feature_list: "list[np.ndarray[np.float32]]") -> "list[np.ndarray[np.float32]]":
  feature_norm_list = feature_list.copy()
  max_list = [max_item(feature_list, index) for index in range(len(get_feature_name_list()))]
  min_list = [min_item(feature_list, index) for index in range(len(get_feature_name_list()))]
  for i, feature in enumerate(feature_list):
    for j, one_feature in enumerate(feature):
      feature_norm_list[i][j] = (one_feature - min_list[j]) / (max_list[j] - min_list[j]) if max_list[j] != min_list[j] else 0
  return feature_norm_list

def standardization(feature_list: "list[np.ndarray[np.float32]]") -> "list[np.ndarray[np.float32]]":
  feature_norm_list = feature_list.copy()
  mean_list = np.mean(feature_list, axis=0)
  std_list = np.std(feature_list, axis=0)
  for i, feature in enumerate(feature_list):
    for j, one_feature in enumerate(feature):
      feature_norm_list[i][j] = (one_feature - mean_list[j]) / std_list[j]
  return feature_norm_list

def max_item(feature_list: "list[np.ndarray[np.float32]]", feature_index: int) -> np.float32:
  max_item = feature_list[0][feature_index]
  for feature in feature_list:
    if max_item < feature[feature_index]:
      max_item = feature[feature_index]
  return max_item

def min_item(feature_list: "list[np.ndarray[np.float32]]", feature_index: int) -> np.float32:
  mix_item = feature_list[0][feature_index]
  for feature in feature_list:
    if mix_item > feature[feature_index]:
      mix_item = feature[feature_index]
  return mix_item