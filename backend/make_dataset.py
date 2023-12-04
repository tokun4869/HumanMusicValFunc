import numpy as np
import csv

from module.io import get_file_name_list, load_sound_list, get_new_file_path
from module.transform import data_augmentation
from module.feature import music2melspectrogram, music2feature, feature2represent
from module.const import *

if __name__ == "__main__":
  print("#01 | === load file and set const ===")
  input_dir = f"{MUSIC_ROOT}/{DATASET_TYPE}/{MODE}_{INPUT_KEY}"
  listen_dir = f"{MUSIC_ROOT}/{DATASET_TYPE}/{MODE}_{LISTEN_KEY}"
  file_name_list = get_file_name_list(listen_dir)
  file_list = load_sound_list(file_name_list)
  num_aug = 3
  file_ext = ".npy"
  list_ext = ".csv"
  dataset_dir = f"{DATASET_ROOT}/{DATASET_TYPE}/{MODE}"
  spec_dataset_base = "spec_dataset"
  feat_dataset_base = "feat_dataset"
  repr_dataset_base = "repr_dataset"
  
  print("#02 | === make crop file ===")
  spec_file_name_list = []
  feat_file_name_list = []
  repr_file_name_list = []
  from_file_name_list = []
  
  for file_name, file in zip(file_name_list, file_list):
    for index in range(num_aug):
      print("    | {} [{}/{}]".format(file_name, index+1, num_aug))
      from_file_name_list.append(file_name)
      new_file = data_augmentation(file)
      file_base = file_name[len(f"{listen_dir}/"):-len(file_ext)]

      spec_file_base = file_base + "_spec"
      new_spec_file_name = get_new_file_path(input_dir, spec_file_base, file_ext)
      print("    | -> {}".format(new_spec_file_name))
      new_spec = music2melspectrogram(new_file)
      np.save(new_spec_file_name, new_spec)
      spec_file_name_list.append(new_spec_file_name)

      feat_file_base = file_base + "_feat"
      new_feat_file_name = get_new_file_path(input_dir, feat_file_base, file_ext)
      print("    | -> {}".format(new_feat_file_name))
      new_feat = music2feature(new_file)
      np.save(new_feat_file_name, new_feat)
      feat_file_name_list.append(new_feat_file_name)

      repr_file_base = file_base + "_repr"
      new_repr_file_name = get_new_file_path(input_dir, repr_file_base, file_ext)
      print("    | -> {}".format(new_repr_file_name))
      new_repr = feature2represent(new_feat)
      np.save(new_repr_file_name, new_repr)
      repr_file_name_list.append(new_repr_file_name)
  
  print("#03 | === make csv load file ===")
  print("    | spec dataset")
  with open(get_new_file_path(dataset_dir, spec_dataset_base, list_ext), "w") as f:
    writer = csv.writer(f)
    writer.writerows(zip(spec_file_name_list, from_file_name_list))
  
  print("    | feat dataset")
  with open(get_new_file_path(dataset_dir, feat_dataset_base, list_ext), "w") as f:
    writer = csv.writer(f)
    writer.writerows(zip(feat_file_name_list, from_file_name_list))

  print("    | repr dataset")
  with open(get_new_file_path(dataset_dir, repr_dataset_base, list_ext), "w") as f:
    writer = csv.writer(f)
    writer.writerows(zip(repr_file_name_list, from_file_name_list))