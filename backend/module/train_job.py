from pydantic import BaseModel
from module.io import get_file_name_list, get_dataset_path
from module.operation import train_operation
from module.const import *

class TrainJob(BaseModel):
  status: str = STATUS_BEFORE
  now_epoch: int = 0
  model_path: str = None
  error: str = None

  def initialize(self) -> None:
    self.status = STATUS_BEFORE
    self.now_epoch = 0
    self.model_path = None
    self.error = None
    
  def __call__(self, rank_list: str) -> None:
    try:
      self.status = STATUS_INPROGRESS
      dataset_path = get_dataset_path()
      file_name_list = get_file_name_list(f"{MUSIC_ROOT}/{DATASET_TYPE}/{MODE}_{LISTEN_KEY}")
      train_operation(dataset_path, rank_list, file_name_list, self.set_now_epoch, self.set_status, self.set_model_path)
    except Exception as e:
      self.status = STATUS_ERROR
      self.error = str(e)
  
  def set_status(self, status: bool):
    self.status = status
  
  def get_status(self) -> bool:
    return self.status
  
  def set_now_epoch(self, now_epoch: int) -> None:
    self.now_epoch = now_epoch

  def get_now_epoch(self) -> int:
    return NUM_EPOCHS
  
  def set_model_path(self, model_path: str) -> None:
    self.model_path = model_path
  
  def get_model_path(self) -> str:
    return self.model_path

  def get_error(self) -> str:
    return self.error
