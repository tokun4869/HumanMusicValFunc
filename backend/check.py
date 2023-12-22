import time
from torch.utils.data import DataLoader
from module.io import get_file_name_list, load_dataset
from module.dataset import TrainDataset
from module.const import *
import module.features as f

if __name__ == "__main__":
    dataset_path = f"{DATASET_ROOT}/{DATASET_TYPE}/{MODE}/{WAVE_DATASET_BASE}{LIST_EXT}"
    rank_list = [1, 10, 6, 8, 9, 2, 4, 5, 3, 7]
    file_name_list = get_file_name_list(f"{MUSIC_ROOT}/{DATASET_TYPE}/{MODE}_{LISTEN_KEY}")
    input_list, target_list = load_dataset(dataset_path, rank_list, file_name_list)
    dataset = TrainDataset(input_list, target_list)
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)
    kernel, fft_win_length, fft_hop_length = f.kernel_matrix()
    for feature, value in dataloader:
        start_time = time.time()
        print(f.music2represent(feature).shape)
        end_time = time.time()
        print(end_time - start_time)
