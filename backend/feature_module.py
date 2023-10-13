# ===== ===== ===== =====
# Music -> Feature Module
# ===== ===== ===== =====

import numpy as np
import librosa

from static_value import SAMPLE_RATE

def melspectrogram(y: "np.ndarray[np.float32]", n_mels: int=1025) -> "np.ndarray[np.float32]":
  return librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=n_mels)

# テンポ
def bpm(y: "np.ndarray[np.float32]") -> np.float32:
  onset_env = librosa.onset.onset_strength(y=y, sr=SAMPLE_RATE)
  return librosa.beat.tempo(onset_envelope=onset_env, sr=SAMPLE_RATE)

# 音圧
def rms(S: "np.ndarray[np.float32]") -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  feature = librosa.feature.rms(S=S)
  feature_mean = feature.mean(axis=1)
  feature_std = np.std(feature, axis=1)
  return feature_mean, feature_std

# 音色
def mfcc(S: "np.ndarray[np.float32]", n_mfcc: int=12, dct_type: int=3) -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  feature = librosa.feature.mfcc(S=S, sr=SAMPLE_RATE, n_mfcc=n_mfcc, dct_type=dct_type)
  feature_mean = feature.mean(axis=1)
  feature_std = np.std(feature, axis=1)
  return feature_mean, feature_std

# スペクトル重心
def centroid(S: "np.ndarray[np.float32]") -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  feature = librosa.feature.spectral_centroid(S=S, sr=SAMPLE_RATE)
  feature_mean = feature.mean(axis=1)
  feature_std = np.std(feature, axis=1)
  return feature_mean, feature_std

# スペクトル範囲
def bandwidth(S: "np.ndarray[np.float32]") -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  feature = librosa.feature.spectral_bandwidth(S=S, sr=SAMPLE_RATE)
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

def title2feature(path: str) -> "np.ndarray[np.float32]":
  y, _ = librosa.load(path, sr=SAMPLE_RATE)
  return music2feature(y)

def music2feature(y: "np.ndarray[np.float32]") -> "np.ndarray[np.float32]":
  feature_array = []

  S = melspectrogram(y)
  rms_mean, rms_std = rms(S)
  mfcc_mean, mfcc_std = mfcc(S)
  centroid_mean, centroid_std = centroid(S)
  bandwidth_mean, bandwidth_std = bandwidth(S)
  tonnetz_mean, tonnetz_std = tonnetz(y)
  zero_crossing_rate_mean, zero_crossing_rate_std = zero_crossing_rate(y)
  feature_array.extend(rms_mean.tolist())
  feature_array.extend(rms_std.tolist())
  feature_array.extend(mfcc_mean.tolist())
  feature_array.extend(mfcc_std.tolist())
  feature_array.extend(centroid_mean.tolist())
  feature_array.extend(centroid_std.tolist())
  feature_array.extend(bandwidth_mean.tolist())
  feature_array.extend(bandwidth_std.tolist())
  feature_array.extend(tonnetz_mean.tolist())
  feature_array.extend(tonnetz_std.tolist())
  feature_array.extend(zero_crossing_rate_mean.tolist())
  feature_array.extend(zero_crossing_rate_std.tolist())
  return np.array(feature_array)