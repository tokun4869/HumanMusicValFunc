import time
import torch
from torch.utils.data import DataLoader
from module.io import get_file_name_list, load_dataset
from module.dataset import TrainDataset
from module.const import *
import module.features as f

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_path = f"{DATASET_ROOT}/{DATASET_MAESTRO}/{MODE_TRAIN}/{WAVE_DATASET_BASE}_0{LIST_EXT}"
    rank_list = [1, 10, 6, 8, 9, 2, 4, 5, 3, 7]
    file_name_list = get_file_name_list(f"{MUSIC_ROOT}/{DATASET_MAESTRO}/{MODE_TRAIN}_{LISTEN_KEY}")
    input_list, _ = load_dataset(dataset_path, rank_list, file_name_list)
    feature = torch.Tensor(input_list[0])
    feature = feature.unsqueeze(0)
    feature = feature.to(device=device)
    
    start_time = time.time()
    tempogram = f.calc_tempogram(feature, device=device)
    end_time = time.time()
    tempogram_time = end_time - start_time
    print(f"tempogram: {tempogram_time:.4}")
    
    start_time = time.time()
    rms = f.calc_rms(feature, device=device)
    end_time = time.time()
    rms_time = end_time - start_time
    print(f"rms: {rms_time:.4}")

    start_time = time.time()
    mfcc = f.calc_mfcc(feature, device=device)
    end_time = time.time()
    mfcc_time = end_time - start_time
    print(f"mfcc: {mfcc_time:.4}")

    start_time = time.time()
    centroid = f.calc_centroid(feature, device=device)
    end_time = time.time()
    centroid_time = end_time - start_time
    print(f"centroid: {centroid_time:.5}")

    start_time = time.time()
    zcr = f.calc_zcr(feature, device=device)
    end_time = time.time()
    zcr_time = end_time - start_time
    print(f"zcr: {zcr_time:.4}")

    start_time = time.time()
    tempogram_mean = torch.mean(tempogram, dim=-1)
    rms_mean = torch.mean(rms, dim=-1)
    mfcc_mean = torch.mean(mfcc, dim=-1)
    centroid_mean = torch.mean(centroid, dim=-1)
    zcr_mean = torch.mean(zcr, dim=-1)
    tempogram_var = torch.var(tempogram, dim=-1)
    rms_var = torch.var(rms, dim=-1)
    mfcc_var = torch.var(mfcc, dim=-1)
    centroid_var = torch.var(centroid, dim=-1)
    zcr_var = torch.var(zcr, dim=-1)
    end_time = time.time()
    mean_var_time = end_time - start_time
    print(f"mean_var: {mean_var_time:.4}")

    start_time = time.time()
    feature_tensor = torch.cat([tempogram_mean, rms_mean, mfcc_mean, centroid_mean, zcr_mean, tempogram_var, rms_var, mfcc_var, centroid_var, zcr_var], dim=1)
    end_time = time.time()
    concat_time = end_time - start_time
    print(f"concat: {concat_time:.4}")

    start_time = time.time()
    feature_tensor = f.calc_melspectrogram(feature, device=device)
    end_time = time.time()
    melspectrogram_time = end_time - start_time
    print(f"melspectrogram: {melspectrogram_time:.4}")

    start_time = time.time()
    repr = f.music2represent(feature, device=device)
    end_time = time.time()
    repr_time = end_time - start_time
    print(f"repr: {repr_time:.4}")

    print(f"repr vs spec: {repr_time / melspectrogram_time:.4}")

    