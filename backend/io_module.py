import numpy as np
import librosa
import glob
import re
import csv

from transform_module import data_augmentation
from feature_module import music2feature, normalization
from static_value import *

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

def load_dataset(dataset_path: str, answer_list: "list[int]") -> "tuple[list[np.ndarray[np.float32]], list[int]]":
  input_list = []
  target_list = []
  file_name_list = get_file_name_list(MUSIC_ROOT + INPUT_DIR)
  with open(dataset_path, "r") as f:
    reader = csv.reader(f)
    for row in reader:
      # input_list.append(music2feature(data_augmentation(load(row[0]))))
      input_list.append(np.load(row[0]))
      target_list.append(answer_list[file_name_list.index(row[1])])
  input_list = normalization(input_list)
  return input_list, target_list