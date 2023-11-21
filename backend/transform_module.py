import numpy as np
import random

from static_value import *

random.seed(SEED)

def data_augmentation(y: "np.ndarray[np.float32]") -> "np.ndarray[np.float32]":
  y = __crop(y)
  return y

def __crop(y: "np.ndarray[np.float32]") -> "np.ndarray[np.float32]":
  sample_length = SAMPLE_RATE*LENGTH
  if len(y) < sample_length:
    y = np.append(y, [0 for _ in range(sample_length-len(y))])
    return y
  l = random.randint(0, len(y)-sample_length)
  r = l + sample_length
  return y[l:r]