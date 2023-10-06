# ===== ===== ===== =====
# Music -> Feature Module
# ===== ===== ===== =====

import numpy as np
import librosa

from static_value import SAMPLE_RATE

# テンポ
def bpm(y: "np.ndarray[np.float32]") -> np.float32:
  onset_env = librosa.onset.onset_strength(y=y, sr=SAMPLE_RATE)
  return librosa.beat.tempo(onset_envelope=onset_env, sr=SAMPLE_RATE)

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

# スペクトル重心
def centroid(y: "np.ndarray[np.float32]") -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  feature = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE)
  feature_mean = feature.mean(axis=1)
  feature_std = np.std(feature, axis=1)
  return feature_mean, feature_std

# スペクトル範囲
def bandwidth(y: "np.ndarray[np.float32]") -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  feature = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE)
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

  for index in range(3):
    l = SAMPLE_RATE * 10 * index
    r = l + SAMPLE_RATE * 10 * (index + 1)
    h = y[l:r]
    rms_mean, rms_std = rms(h)
    mfcc_mean, mfcc_std = mfcc(h)
    centroid_mean, centroid_std = centroid(h)
    bandwidth_mean, bandwidth_std = bandwidth(h)
    tonnetz_mean, tonnetz_std = tonnetz(h)
    zero_crossing_rate_mean, zero_crossing_rate_std = zero_crossing_rate(h)
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