import numpy as np

from pydantic import BaseModel
from feature import music2feature
from transform import title2augmusic
from io_module import get_new_file_path
from static_value import *

class MakeDatasetJob(BaseModel):
  n_aug: int = 10
  sr: int = 48000
  status: str = STATUS_BEFORE
  now_file_name: str = None
  now_progress: int = None
  train_dataset_dir: str = None
  target_dataset_dir: str = None
  error: str = None
    
  def __call__(self, file_name_list: "list[str]", answer_list: "list[int]") -> None:
    try:
      self.status = STATUS_INPROGRESS
      self.dataset(file_name_list, answer_list)
    except Exception as e:
      self.status = STATUS_ERROR
      self.error = str(e)

  def get_status(self) -> str:
    return self.status
  
  def get_error(self) -> str:
    return self.error

  def get_progress(self) -> "tuple[str, float]":
    return self.now_file_name, self.now_progress
  
  def get_dataset_dir(self) -> "tuple[str, str]":
    return self.train_dataset_dir, self.target_dataset_dir

  def dataset(self, file_name_list: "list[str]", answer_list: "list[int]") -> None:
    train = np.array([])
    target = np.array([])
    for file_name, answer in zip(file_name_list, answer_list):
      self.now_file_name = file_name
      for index in range(self.n_aug):
        self.now_progress = index / self.n_aug * 100
        if train.size == 0:
          train = music2feature(title2augmusic(file_name, self.sr), self.sr)
        else:
          train = np.vstack([train, music2feature(title2augmusic(file_name, self.sr), self.sr)])
        if target.size == 0:
          target = np.array([len(answer_list) - answer])
        else:
          target = np.vstack([target, len(answer_list) - answer])
    self.train_dataset_dir = get_new_file_path(DATASET_ROOT, TRAIN_BASE, ".npy")
    self.target_dataset_dir = get_new_file_path(DATASET_ROOT, TARGET_BASE, ".npy")
    np.save(self.train_dataset_dir, train)
    np.save(self.target_dataset_dir, target)
    self.status = STATUS_FINISH
