import numpy as np
import librosa
import glob
import re

from static_value import SAMPLE_RATE

def load(path: str) -> "np.ndarray[np.float32]":
  y, _ = librosa.load(path, sr=SAMPLE_RATE)
  return y

def __atoi(text: str):
  return int(text) if text.isdigit() else text

def __natural_keys(text: str):
  return [__atoi(c) for c in re.split(r"(\d+)", text)]

def get_file_name_list(dir: str) -> "list[str]":
  return sorted(glob.glob(dir + "*.mp3"), key=__natural_keys)

def get_file_list(file_name_list: "list[str]") -> "list[np.ndarray[np.float32]]":
  file_list = []
  for file_name in file_name_list:
    file_list.append(load(file_name))
  return file_list

def get_new_file_path(dir: str, base: str, ext: str) -> str:
  file_name_list = glob.glob("{}{}_*{}".format(dir, base, ext))
  max_index = -1
  for file_name in file_name_list:
    index = int(file_name[len(dir)+len(base)+len("_") : -len(ext)])
    if max_index < index:
      max_index = index
  path = "{}{}_{}{}".format(dir, base, max_index+1, ext)
  return path