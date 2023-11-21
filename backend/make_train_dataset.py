import numpy as np
import csv

from io_module import get_file_name_list, get_file_list, get_new_file_path
from transform_module import data_augmentation
from feature_module import melspectrogram, music2feature
from static_value import *

if __name__ == "__main__":
  print("#01 | === load file and set const ===")
  file_name_list = get_file_name_list(MUSIC_ROOT + TRAIN_LISTEN_DIR)
  file_list = get_file_list(file_name_list)
  num_aug = 3
  file_ext = ".npy"
  list_ext = ".csv"
  wave_dataset_base = "train_wave_dataset"
  spec_dataset_base = "train_spec_dataset"
  feat_dataset_base = "train_feat_dataset"
  
  print("#02 | === make crop file ===")
  wave_file_name_list = []
  spec_file_name_list = []
  feat_file_name_list = []
  from_file_name_list = []
  
  for file_name, file in zip(file_name_list, file_list):
    for index in range(num_aug):
      print("    | {} [{}/{}]".format(file_name, index+1, num_aug))
      from_file_name_list.append(file_name)
      new_file = data_augmentation(file)

      new_wave_file_name = MUSIC_ROOT + TRAIN_INPUT_DIR + file_name[len(MUSIC_ROOT + TRAIN_LISTEN_DIR):-len(file_ext)] + "_{}_wave{}".format(index, file_ext)
      print("    | -> {}".format(new_wave_file_name))
      np.save(new_wave_file_name, new_file)
      wave_file_name_list.append(new_wave_file_name)

      new_spec_file_name = MUSIC_ROOT + TRAIN_INPUT_DIR + file_name[len(MUSIC_ROOT + TRAIN_LISTEN_DIR):-len(file_ext)] + "_{}_spec{}".format(index, file_ext)
      print("    | -> {}".format(new_spec_file_name))
      new_spec = melspectrogram(new_file)
      np.save(new_spec_file_name, new_spec)
      spec_file_name_list.append(new_spec_file_name)

      new_feat_file_name = MUSIC_ROOT + TRAIN_INPUT_DIR + file_name[len(MUSIC_ROOT + TRAIN_LISTEN_DIR):-len(file_ext)] + "_{}_feat{}".format(index, file_ext)
      print("    | -> {}".format(new_feat_file_name))
      new_feat = music2feature(new_file)
      np.save(new_feat_file_name, new_feat)
      feat_file_name_list.append(new_feat_file_name)
  
  print("#03 | === make csv load file ===")
  print("    | wave dataset")
  with open(get_new_file_path(DATASET_ROOT, wave_dataset_base, list_ext), "w") as f:
    writer = csv.writer(f)
    writer.writerows(zip(wave_file_name_list, from_file_name_list))

  print("    | spec dataset")
  with open(get_new_file_path(DATASET_ROOT, spec_dataset_base, list_ext), "w") as f:
    writer = csv.writer(f)
    writer.writerows(zip(spec_file_name_list, from_file_name_list))
  
  print("    | feat dataset")
  with open(get_new_file_path(DATASET_ROOT, feat_dataset_base, list_ext), "w") as f:
    writer = csv.writer(f)
    writer.writerows(zip(feat_file_name_list, from_file_name_list))