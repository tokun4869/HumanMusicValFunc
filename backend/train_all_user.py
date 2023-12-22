from module.io import get_dataset_path, load_all_user_data, get_file_name_list
from module.operation import train_operation
from module.const import *


if __name__ == "__main__":
  dataset_path = get_dataset_path()
  rank_list_list = load_all_user_data()
  file_name_list = get_file_name_list(f"{MUSIC_ROOT}/{DATASET_TYPE}/{MODE}_{LISTEN_KEY}")
  for rank_list in rank_list_list:
    train_operation(dataset_path, rank_list, file_name_list)