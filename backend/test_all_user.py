import sys
from module.io import get_file_name_list, load_sound_list, load_all_user_data
from module.operation import test_operation
from module.util import torch_fix_seed
from module.const import *


if __name__ == "__main__":
  torch_fix_seed()
  model_path_list = get_file_name_list(f"{MODEL_ROOT}")
  model_path_list = [EXTRACTOR_TYPE in model_path_list]
  file_name_list = get_file_name_list(f"{MUSIC_ROOT}/{DATASET_TYPE}/{MODE}_{LISTEN_KEY}")
  sound_list = load_sound_list(file_name_list)
  target_list_list, label_list = load_all_user_data()
  for target_list, label in zip(target_list_list, label_list):
    model_path = f"{MODEL_ROOT}/model_{label}_0{MODEL_EXT}"
    test_operation(model_path, sound_list, target_list, label)