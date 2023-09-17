import numpy as np
from pydantic import BaseModel
from feature import music2feature
from transform import title2augmusic

class MakeDatasetJob(BaseModel):
  n_aug: int = 5
  sr: int = 48000
  status: bool = False
  now_file_name: str = None
  now_progress: int = None
  train_dataset_dir: str = None
  target_dataset_dir: str = None
    
  def __call__(self, file_name_list: "list[str]", answer_list: "list[int]") -> None:
    try:
      self.status = True
      self.dataset(file_name_list, answer_list)
    except Exception as e:
      print(e)
    finally:
      self.status = False
  
  def get_status(self) -> bool:
    return self.status

  def get_progress(self) -> "tuple[str, float]":
    return self.now_file_name, self.now_progress

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
          target = np.array([answer])
        else:
          target = np.vstack([target, answer])
    np.save("data/aug_music/train", train)
    np.save("data/aug_music/target", target)
