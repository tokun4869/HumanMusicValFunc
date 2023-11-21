import csv
import numpy as np
from io_module import get_file_name_list, get_file_list, get_new_file_path
from transform_module import inverse, gain, time_shift, pitch_shift, white_noise
from feature_module import music2feature, get_feature_name_list
from static_value import *

def normalization(feature_list: "list[np.ndarray[np.float32]]") -> "list[np.ndarray[np.float32]]":
  feature_norm_list = feature_list.copy()
  max_list = [max_item(feature_list, index) for index in range(len(get_feature_name_list()))]
  min_list = [min_item(feature_list, index) for index in range(len(get_feature_name_list()))]
  for i, feature in enumerate(feature_list):
    for j, one_feature in enumerate(feature):
      feature_norm_list[i][j] = (one_feature - min_list[j]) / (max_list[j] - min_list[j]) if max_list[j] != min_list[j] else 0
  return feature_norm_list

def max_item(feature_list: "list[np.ndarray[np.float32]]", feature_index: int) -> np.float32:
  max_item = feature_list[0][feature_index]
  for feature in feature_list:
    if max_item < feature[feature_index]:
      max_item = feature[feature_index]
  return max_item

def min_item(feature_list: "list[np.ndarray[np.float32]]", feature_index: int) -> np.float32:
  mix_item = feature_list[0][feature_index]
  for feature in feature_list:
    if mix_item > feature[feature_index]:
      mix_item = feature[feature_index]
  return mix_item

if __name__ == "__main__":
  print("=== feature argment test ===")
  print("#01 | data_load")
  file_name_list = get_file_name_list(MUSIC_ROOT + INPUT_DIR)
  file_list = get_file_list(file_name_list)
  data_augmentation_list = [inverse, gain, time_shift, pitch_shift, white_noise]
  feature_dif_list = []

  print("#02 | feature_dif_calc")
  for data_augmentation in data_augmentation_list:
    print("    | {}". format(data_augmentation.__name__))
    for name, file in zip(file_name_list, file_list):
      for index in range(3):
        print("    | {} [{}/3]".format(name, index+1))
        feature_no_arg = music2feature(file)
        if data_augmentation.__name__ == "inverse":
          feature_arg = music2feature(data_augmentation(file))
        elif data_augmentation.__name__ == "gain":
          feature_arg = music2feature(data_augmentation(file, 0.1))
        elif data_augmentation.__name__ == "time_shift":
          feature_arg = music2feature(data_augmentation(file, len(file)))
        elif data_augmentation.__name__ == "pitch_shift":
          feature_arg = music2feature(data_augmentation(file, 0.2))
        elif data_augmentation.__name__ == "white_noise":
          feature_arg = music2feature(data_augmentation(file, 0.02))
        feature_dif = feature_arg - feature_no_arg
        feature_dif_list.append(feature_dif)
    feature_norm_dif_list = normalization(feature_dif_list)
    feature_mean_dif_list = np.mean(feature_norm_dif_list, axis=0)
    
    print("#03 | dif_mean_csv_save")
    with open(get_new_file_path(FEATURE_ROOT, "feature_dif_mean", ".csv"), "w") as f:
      writer = csv.writer(f)
      writer.writerow(get_feature_name_list())
      writer.writerow(feature_mean_dif_list)
    
        
    