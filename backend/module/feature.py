# ===== ===== ===== =====
# Music -> Feature Module
# ===== ===== ===== =====

import numpy as np
import librosa
from module.transform import data_augmentation
from module.const import *


def tempogram(y: np.ndarray[np.float32], win_length: int=10) -> np.ndarray[np.float32]:
  """
  音楽から各時刻でのテンポの推定をおこなう
  
  Parameters
  ----------
  y : np.ndarray[np.float32]
    対象の音楽
  
  Returns
  ----------
  feature : np.ndarray[np.float32]
    変換後の特徴量
  """

  return librosa.feature.tempogram(y=y, sr=SAMPLE_RATE, win_length=win_length)


# 音圧
def rms(y: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
  return librosa.feature.rms(y=y)

# 音色
def mfcc(y: np.ndarray[np.float32], n_mfcc: int=12, dct_type: int=3) -> np.ndarray[np.float32]:
  return librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=n_mfcc, dct_type=dct_type)

# 和音に関する特徴量
def tonnetz(y: "np.ndarray[np.float32]") -> np.ndarray[np.float32]:
  return librosa.feature.tonnetz(y=y, sr=SAMPLE_RATE)

# ゼロ交叉率
def zero_crossing_rate(y: "np.ndarray[np.float32]") -> np.ndarray[np.float32]:
  return librosa.feature.zero_crossing_rate(y=y)

# 特徴量の平均の計算
def feature_mean(feature: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
  return np.mean(feature, axis=1)

# 特徴量の分散の計算
def feature_var(feature: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
  return np.var(feature, axis=1)

# 各特徴量の計算
def music2feature(y: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
  feature_list = []
  feature_list.extend(tempogram(y).tolist())
  feature_list.extend(rms(y).tolist())
  feature_list.extend(mfcc(y).tolist())
  feature_list.extend(tonnetz(y).tolist())
  feature_list.extend(zero_crossing_rate(y).tolist())
  return np.array(feature_list)

def feature2represent(feature_array: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
  feature_mean_array = feature_mean(feature_array)
  feature_var_array = feature_var(feature_array)
  represent_array = np.concatenate([feature_mean_array, feature_var_array])
  return represent_array

def music2represent(y: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
  feature_array = music2feature(y)
  represent_array = feature2represent(feature_array)
  return represent_array

def music2melspectrogram(y: np.ndarray[np.float32]) -> np.ndarray[np.ndarray[np.float32]]:
  return librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE)

def musics2input(music_list: list[np.ndarray[np.float32]]) -> list[np.ndarray[np.float32]]:
  if MODEL_TYPE == MODEL_SPEC:
    return [music2melspectrogram(data_augmentation(y)) for y in music_list]
  if MODEL_TYPE == MODEL_FEAT:
    return [music2feature(data_augmentation(y)) for y in music_list]
  if True:
    return [music2represent(data_augmentation(y)) for y in music_list]