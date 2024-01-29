import csv
from module.io import get_new_file_path, load_all_user_data, get_file_name_list, load_sound_list
from module.operation import test_operation
from module.const import *


if __name__ == "__main__":
  train_dataset_type_list = [DATASET_MAESTRO, DATASET_MTG, DATASET_MUSICNET]
  retrain_dataset_type_list = [DATASET_MAESTRO, DATASET_MTG, DATASET_MUSICNET]
  test_dataset_type_list = [DATASET_MAESTRO, DATASET_MTG, DATASET_MUSICNET]
  extractor_type_list = [EXTRACTOR_REPR, EXTRACTOR_SPEC, EXTRACTOR_FEAT]
  output_line = []
  target_line = []
  time_line = []
  
  for train_dataset_type in train_dataset_type_list:
    for retrain_dataset_type in retrain_dataset_type_list:
      for test_dataset_type in test_dataset_type_list:
        target_list, user_list = load_all_user_data(test_dataset_type, mode=MODE_RETEST)
        file_name_list = get_file_name_list(f"{MUSIC_ROOT}/{test_dataset_type}/{MODE_RETEST}_{LISTEN_KEY}")
        sound_list = load_sound_list(file_name_list)
        for extractor_type in extractor_type_list:
          output_line.append([train_dataset_type, retrain_dataset_type, test_dataset_type, extractor_type])
          target_line.append([train_dataset_type, retrain_dataset_type, test_dataset_type, extractor_type])
          time_line.append([train_dataset_type, retrain_dataset_type, test_dataset_type, extractor_type])
          for target, user in zip(target_list, user_list):
            label = f"RETRAIN_{train_dataset_type}_{retrain_dataset_type}_{extractor_type}_{user}"
            model_path = f"{MODEL_ROOT}/model_{label}_0{MODEL_EXT}"
            output_list, time_list = test_operation(model_path=model_path, sound_list=sound_list, label=label, extractor=extractor_type)
            output_line.append(output_list)
            target_line.append(target)
            time_line.append(time_list)

  print("#03 | === SAVE RESULT ===")
  file_base = f"output_RETEST"
  with open(get_new_file_path(RESULT_ROOT, file_base, LIST_EXT), "w") as f:
    writer = csv.writer(f)
    writer.writerows(output_line)
  
  file_base = f"target_RETEST"
  with open(get_new_file_path(RESULT_ROOT, file_base, LIST_EXT), "w") as f:
    writer = csv.writer(f)
    writer.writerows(target_line)
  
  file_base = f"time_RETEST"
  with open(get_new_file_path(TIME_ROOT, file_base, LIST_EXT), "w") as f:
    writer = csv.writer(f)
    writer.writerows(time_line)