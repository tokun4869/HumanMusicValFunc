import numpy as np
import librosa
import random

def title2augmusic(path: str, sr: int) -> "np.ndarray[np.float32]":
  y, _ = librosa.load(path, sr=sr)
  y = gain(y, 0.1)
  y = time_shift(y, len(y))
  y = pitch_shift(y, sr, 0.2)
  y = white_noise(y, 0.02)
  return y

def gain(y: "np.ndarray[np.float32]", rate: float) -> "np.ndarray[np.float32]":
  return y * random.uniform(1-rate, 1+rate)

def time_shift(y: "np.ndarray[np.float32]", shift: int) -> "np.ndarray[np.float32]":
  return np.roll(y, random.randint(0, shift))

def pitch_shift(y: "np.ndarray[np.float32]", sr: int, shift: int) -> "np.ndarray[np.float32]":
  return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=random.uniform(-shift, shift), bins_per_octave=12)

def white_noise(y: "np.ndarray[np.float32]", rate: float) -> "np.ndarray[np.float32]":
  return y + random.uniform(0, rate) * np.random.randn(len(y))