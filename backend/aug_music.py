import glob
import numpy as np

from feature import title2feature, music2feature
from transform import title2augmusic

def music_list(dir: str):
  return glob.glob(dir + "/*.wav")

def feature_list(file_name_list: "list[str]", sr: int):
  return np.array([title2feature(dir + file_name, sr) for file_name in file_name_list])

def answer_list(file_name_list: "list[str]"):
  array = np.array([0.0 for _ in range(len(file_name_list))])
  for index, file_name in enumerate(file_name_list):
    print("=== {} の評価 ===".format(file_name))
    print("# 何番目に好みか?")
    while(True):
      tmp_input = input()
      if tmp_input.isdecimal() and int(tmp_input) >= 1 and int(tmp_input) <= len(file_name_list):
        break
      print(": 1 <= x <= {} の整数を入力してください".format(len(file_name_list)))
    array[index] = len(file_name_list) - int(tmp_input)
  return array

def dataset(file_name_list: "list[str]", answer_list: "list[int]"):
  n_aug = 5
  sr = 48000
  train = np.array([])
  target = np.array([])
  answer_list = len(answer_list) - answer_list
  for file_name, answer in zip(file_name_list, answer_list):
    for _ in range(n_aug):
      if train.size == 0:
        train = music2feature(title2augmusic(file_name, sr), sr)
      else:
        train = np.vstack([train, music2feature(title2augmusic(file_name, sr), sr)])
      if target.size == 0:
        target = np.array([answer])
      else:
        target = np.vstack([target, answer])
  return train, target

def data_augmentation(file_name_list: "list[str]", sr: int, n_aug: int) -> "tuple(np.ndarray[np.float32], np.ndarray[np.float32])":
  print("=== Start Augmentation! ===")
  train = np.array([])
  target = np.array([])
  input_answer_list = answer_list(file_name_list)
  for file_name, answer in zip(file_name_list, input_answer_list):
    for index in range(n_aug):
      print("File name: {} [{:3d}/{:3d}]".format(file_name, index + 1, n_aug))
      if train.size == 0:
        train = music2feature(title2augmusic(file_name, sr), sr)
      else:
        train = np.vstack([train, music2feature(title2augmusic(file_name, sr), sr)])
      if target.size == 0:
        target = answer
      else:
        target = np.vstack([target, answer])
  return train, target

if __name__ == "__main__":
  file_name_list = music_list("data/music/train")
  train_array, target_array = data_augmentation(file_name_list, 48000, 10)
  np.save("data/aug_music/train", train_array)
  np.save("data/aug_music/target", target_array)