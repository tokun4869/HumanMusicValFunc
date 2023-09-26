from static_value import *
import numpy as np
import torch

from model import Model
from feature_module import title2feature
from io_module import get_file_name_list

def feature_list(file_name_list: "list[str]", sr: int):
  return np.array([title2feature(file_name, sr) for file_name in file_name_list])

def test_output(model_path: str) -> "dict(str, float)":
  output_list = []
  model = Model()
  model.load_state_dict(torch.load(model_path))
  model.eval()

  file_name_list = get_file_name_list(MUSIC_ROOT + TEST_DIR)
  test_list = feature_list(file_name_list, 48000)
  for _, data in zip(file_name_list, test_list):
    data = torch.from_numpy(data).float()
    output = model(data).item()
    output_list.append(output)
  return output_list