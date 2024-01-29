from module.operation import dataset_operation
from module.util import torch_fix_seed
from module.const import *

if __name__ == "__main__":
  print("#01 | === load file and set const ===")
  torch_fix_seed(1)
  dataset_type_list = [DATASET_MAESTRO, DATASET_MTG, DATASET_MUSICNET]
  mode_list = [MODE_TRAIN, MODE_TEST]
  for dataset_type in dataset_type_list:
    for mode in mode_list:
      dataset_operation(dataset_type=dataset_type, mode=mode)