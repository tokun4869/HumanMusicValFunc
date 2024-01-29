import csv
from module.io import get_new_file_path, load_all_user_data, get_file_name_list, get_model_path_list
from module.operation import retrain_operation
from module.const import *
from module.util import torch_fix_seed


if __name__ == "__main__":
  torch_fix_seed()
  train_dataset_list = [DATASET_MAESTRO, DATASET_MTG, DATASET_MUSICNET]
  test_dataset_list = [DATASET_MAESTRO, DATASET_MTG, DATASET_MUSICNET]
  extractor_list = [EXTRACTOR_REPR, EXTRACTOR_SPEC, EXTRACTOR_FEAT]
  train_loss_list = []
  valid_loss_list = []
  time_list = []
  data_load_time_list = []
  for train_dataset in train_dataset_list:
    for test_dataset in test_dataset_list:
      dataset_path = f"{DATASET_ROOT}/{test_dataset}/{MODE_TEST}/{WAVE_DATASET_BASE}_1{LIST_EXT}"
      rank_list_list, user_list = load_all_user_data(test_dataset, mode=MODE_TEST)
      file_name_list = get_file_name_list(f"{MUSIC_ROOT}/{test_dataset}/{MODE_TEST}_{LISTEN_KEY}")
      for extractor in extractor_list:
        train_loss_list.append([train_dataset, test_dataset, extractor])
        valid_loss_list.append([train_dataset, test_dataset, extractor])
        time_list.append([train_dataset, test_dataset, extractor])
        data_load_time_list.append([train_dataset, test_dataset, extractor])
        model_path_list = get_model_path_list(f"model_{train_dataset}_{extractor}")
        for rank_list, user, model_path in zip(rank_list_list, user_list, model_path_list):
          label = f"RETRAIN_{train_dataset}_{test_dataset}_{extractor}_{user}"
          train_loss, valid_loss, time, data_load_time = retrain_operation(dataset_path, rank_list, file_name_list, model_path, label, extractor)
          train_loss_list.append(train_loss)
          valid_loss_list.append(valid_loss)
          time_list.append(time)
          data_load_time_list.append([data_load_time])
  
  file_base = "loss_RETRAIN_train"
  with open(get_new_file_path(LOSS_ROOT, file_base, LIST_EXT), "w") as f:
    writer = csv.writer(f)
    writer.writerows(train_loss_list)
  
  file_base = "loss_RETRAIN_valid"
  with open(get_new_file_path(LOSS_ROOT, file_base, LIST_EXT), "w") as f:
    writer = csv.writer(f)
    writer.writerows(valid_loss_list)
  
  file_base = "epoch_time_RETRAIN"
  with open(get_new_file_path(TIME_ROOT, file_base, LIST_EXT), "w") as f:
    writer = csv.writer(f)
    writer.writerows(time_list)
  
  file_base = "load_time_RETRAIN"
  with open(get_new_file_path(TIME_ROOT, file_base, LIST_EXT), "w") as f:
    writer = csv.writer(f)
    writer.writerows(data_load_time_list)