from module.io import get_dataset_path, get_file_name_list
from module.operation import train_operation
from module.util import torch_fix_seed
from module.const import *

if __name__ == "__main__":
  torch_fix_seed()
  dataset_path = get_dataset_path()
  dataset_path = f"{DATASET_ROOT}/{DATASET_TYPE}/{MODE}/{WAVE_DATASET_BASE}{LIST_EXT}"
  rank_list = [1, 10, 6, 8, 9, 2, 4, 5, 3, 7]
  file_name_list = get_file_name_list(f"{MUSIC_ROOT}/{DATASET_TYPE}/{MODE}_{LISTEN_KEY}")
  train_operation(dataset_path, rank_list, file_name_list)