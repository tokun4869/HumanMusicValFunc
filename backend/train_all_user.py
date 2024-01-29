import csv
from module.io import get_new_file_path, load_all_user_data, get_file_name_list
from module.operation import train_operation
from module.const import *
from module.util import torch_fix_seed


if __name__ == "__main__":
  torch_fix_seed()
  train_dataset_type_list = [DATASET_MAESTRO, DATASET_MTG, DATASET_MUSICNET]
  retrain_dataset_type_list = [DATASET_MAESTRO, DATASET_MTG, DATASET_MUSICNET]
  test_dataset_type_list = [DATASET_MAESTRO, DATASET_MTG, DATASET_MUSICNET]
  extractor_list = [EXTRACTOR_REPR, EXTRACTOR_SPEC, EXTRACTOR_FEAT]
  train_loss_list = []
  valid_loss_list = []
  time_list = []
  data_load_time_list = []
  for train_dataset in train_dataset_type_list:
    target_list_list, user_list = load_all_user_data(dataset_type=train_dataset, mode=MODE_TRAIN)
    dataset_path = f"{DATASET_ROOT}/{train_dataset}/{MODE_TRAIN}/{WAVE_DATASET_BASE}_2{LIST_EXT}"
    file_name_list = get_file_name_list(f"{MUSIC_ROOT}/{train_dataset}/{MODE_TRAIN}_{LISTEN_KEY}")
    for extractor in extractor_list:
      train_loss_list.append([train_dataset, extractor])
      valid_loss_list.append([train_dataset, extractor])
      time_list.append([train_dataset, extractor])
      data_load_time_list.append([train_dataset, extractor])
      for target_list, user in zip(target_list_list, user_list):
        label = f"{train_dataset}_{extractor}_{user}"
        train_loss, valid_loss, time, data_load_time = train_operation(dataset_path, target_list, file_name_list, label, extractor=extractor)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        time_list.append(time)
        data_load_time_list.append([data_load_time])
  
  file_base = "loss_TRAIN_train"
  with open(get_new_file_path(LOSS_ROOT, file_base, LIST_EXT), "w") as f:
    writer = csv.writer(f)
    writer.writerows(train_loss_list)
  
  file_base = "loss_TRAIN_valid"
  with open(get_new_file_path(LOSS_ROOT, file_base, LIST_EXT), "w") as f:
    writer = csv.writer(f)
    writer.writerows(valid_loss_list)
  
  file_base = "epoch_time_TRAIN"
  with open(get_new_file_path(TIME_ROOT, file_base, LIST_EXT), "w") as f:
    writer = csv.writer(f)
    writer.writerows(time_list)
  
  file_base = "load_time_TRAIN"
  with open(get_new_file_path(TIME_ROOT, file_base, LIST_EXT), "w") as f:
    writer = csv.writer(f)
    writer.writerows(data_load_time_list)
  