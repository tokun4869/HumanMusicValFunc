import sys
from module.io import get_file_name_list, load_sound_list
from module.operation import test_operation
from module.const import *


if __name__ == "__main__":
  model_base = sys.argv[1]
  model_path = f"{MODEL_ROOT}/{model_base}{MODEL_EXT}"
  file_name_list = get_file_name_list(f"{MUSIC_ROOT}/{DATASET_TYPE}/{MODE}_{LISTEN_KEY}")
  sound_list = load_sound_list(file_name_list)
  test_operation(model_path, sound_list)