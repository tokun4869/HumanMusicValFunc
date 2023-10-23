import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from io_module import get_file_name_list, get_file_list, get_new_file_path
from feature_module import music2feature, get_feature_name_list
from static_value import *

def normalization(feature_list: "list[np.ndarray[np.float32]]") -> "list[np.ndarray[np.float32]]":
  feature_norm_list = feature_list.copy()
  max_list = [max_item(feature_list, index) for index in range(len(get_feature_name_list()))]
  min_list = [min_item(feature_list, index) for index in range(len(get_feature_name_list()))]
  for i, feature in enumerate(feature_list):
    for j, one_feature in enumerate(feature):
      feature_norm_list[i][j] = (one_feature - min_list[j]) / (max_list[j] - min_list[j])
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

def draw_heatmap(data: "np.ndarray[np.ndarray[np.float32]]", labels: "list[str]"):
    label = get_feature_name_list()
    df = pd.DataFrame(data=data, index=label, columns=label)
    plt.figure(figsize=(30, 25))
    sns.heatmap(df, annot=True, square=True, cmap="Blues")
    plt.savefig(get_new_file_path(FEATURE_ROOT, "feature_cor", ".png"))

if __name__ == "__main__":
  print("=== feature test ===")
  print("#01 | data_load")
  file_name_list = get_file_name_list(MUSIC_ROOT + TRAIN_DIR)
  file_list = get_file_list(file_name_list)
  print("#02 | feature_calc")
  feature_list = []
  for name, file in zip(file_name_list, file_list):
    print("    | {}".format(name))
    feature = music2feature(file)
    feature_list.append(feature)
  feature_norm_list = normalization(feature_list)
  feature_var_list = np.var(feature_norm_list, axis=0)
  feature_cor_list = np.abs(np.corrcoef(feature_norm_list, rowvar=False))

  print("#03 | save_var_csv")
  with open(get_new_file_path(FEATURE_ROOT, "feature", ".csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(get_feature_name_list())
    writer.writerow(feature_var_list)

  print("#04 | save_cor_heatmap")
  draw_heatmap(feature_cor_list, get_feature_name_list())
