import csv
from static_value import *
import numpy as np
import torch

from feature_module import music2feature, melspectrogram, normalization, standardization
from transform_module import data_augmentation
from io_module import get_file_list, get_file_name_list, get_new_file_path, load_model

def feature_list(file_name_list: "list[str]") -> "np.ndarray[np.ndarray[np.float32]]":
  file_list = get_file_list(file_name_list)
  if IS_FEAT:
    crop_feat_list = [music2feature(data_augmentation(file)) for file in file_list]
    if IS_NORM:
      norm_feat_list = normalization(crop_feat_list)
    elif IS_STD:
      norm_feat_list = standardization(crop_feat_list)
    else:
      norm_feat_list = crop_feat_list
    norm_feat_array = np.array(norm_feat_list)
    return norm_feat_array
  else:
    crop_spec_list = [melspectrogram(data_augmentation(file)) for file in file_list]
    crop_spec_array = np.array(crop_spec_list)
    return crop_spec_array

def test_output(model_path: str) -> "dict(str, float)":
  output_list = []
  model = load_model(model_path)

  file_name_list = get_file_name_list(MUSIC_ROOT + TEST_LISTEN_DIR)
  test_list = feature_list(file_name_list)
  data = torch.from_numpy(test_list).to(torch.float)
  data = torch.unsqueeze(data, dim=1)
  output = model(data)
  output_list = torch.squeeze(output).tolist()
  
  with open(get_new_file_path(RESULT_ROOT, "result", ".csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(output_list)

  return output_list