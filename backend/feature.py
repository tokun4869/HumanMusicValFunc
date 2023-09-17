# ===== ===== ===== =====
# Music -> Feature Module
# ===== ===== ===== =====

import numpy as np
import librosa

# テンポ
def bpm(y: "np.ndarray[np.float32]", sr: int) -> np.float32:
  onset_env = librosa.onset.onset_strength(y=y, sr=sr)
  return librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

# 音圧
def rms(y: "np.ndarray[np.float32]") -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  feature = librosa.feature.rms(y=y)
  feature_mean = feature.mean(axis=1)
  feature_std = np.std(feature, axis=1)
  return feature_mean, feature_std

# 音色
def mfcc(y: "np.ndarray[np.float32]", sr: int, n_mfcc: int=12, dct_type: int=3) -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  feature = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, dct_type=dct_type)
  feature_mean = feature.mean(axis=1)
  feature_std = np.std(feature, axis=1)
  return feature_mean, feature_std

# スペクトル重心
def centroid(y: "np.ndarray[np.float32]", sr: int) -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  feature = librosa.feature.spectral_centroid(y=y, sr=sr)
  feature_mean = feature.mean(axis=1)
  feature_std = np.std(feature, axis=1)
  return feature_mean, feature_std

# スペクトル範囲
def bandwidth(y: "np.ndarray[np.float32]", sr: int) -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  feature = librosa.feature.spectral_bandwidth(y=y, sr=sr)
  feature_mean = feature.mean(axis=1)
  feature_std = np.std(feature, axis=1)
  return feature_mean, feature_std

# 和音に関する特徴量
def tonnetz(y: "np.ndarray[np.float32]", sr: int) -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  feature = librosa.feature.tonnetz(y=y, sr=sr)
  feature_mean = feature.mean(axis=1)
  feature_std = np.std(feature, axis=1)
  return feature_mean, feature_std

# ゼロ交叉率
def zero_crossing_rate(y: "np.ndarray[np.float32]") -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  feature = librosa.feature.zero_crossing_rate(y=y)
  feature_mean = feature.mean(axis=1)
  feature_std = np.std(feature, axis=1)
  return feature_mean, feature_std

def title2feature(path: str, sr: int) -> "np.ndarray[np.float32]":
  y, _ = librosa.load(path, sr=sr)
  return music2feature(y, sr)

def music2feature(y: "np.ndarray[np.float32]", sr: int) -> "np.ndarray[np.float32]":
  rms_mean, rms_std = rms(y)
  mfcc_mean, mfcc_std = mfcc(y, sr)
  centroid_mean, centroid_std = centroid(y, sr)
  bandwidth_mean, bandwidth_std = bandwidth(y, sr)
  tonnetz_mean, tonnetz_std = tonnetz(y, sr)
  zero_crossing_rate_mean, zero_crossing_rate_std = zero_crossing_rate(y)
  feature_array = np.concatenate((rms_mean, rms_std, mfcc_mean, mfcc_std, centroid_mean, centroid_std, bandwidth_mean, bandwidth_std, tonnetz_mean, tonnetz_std, zero_crossing_rate_mean, zero_crossing_rate_std))
  return feature_array