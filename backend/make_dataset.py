import numpy as np
import csv

from io_module import get_file_name_list, get_file_list, get_new_file_path
from transform_module import data_augmentation
from feature_module import music2feature
from static_value import *

if __name__ == "__main__":
  print("=== make dataset ===")
  print("#01 | load file and set const")
  file_name_list = get_file_name_list(MUSIC_ROOT + TRAIN_DIR)
  file_list = get_file_list(file_name_list)
  num_aug = 3
  file_ext = ".npy"
  list_ext = ".csv"
  dataset_base = "dataset"
  
  print("#02 | make aug file")
  new_file_name_list = []
  from_file_name_list = []
  for index in range(num_aug):
    for file_name, file in zip(file_name_list, file_list):
      print("    | {} [{}/{}]".format(file_name, index, num_aug))
      new_file_name =  file_name[:-len(file_ext)] + "_{}{}".format(index, file_ext)
      new_file = data_augmentation(file)
      new_feature = music2feature(new_file)
      np.save(new_file_name, new_feature)
      new_file_name_list.append(new_file_name)
      from_file_name_list.append(file_name)
  
  print("#03 | make csv load file")
  with open(get_new_file_path(DATASET_ROOT, dataset_base, list_ext), "w") as f:
    writer = csv.writer(f)
    writer.writerows(zip(new_file_name_list, from_file_name_list))