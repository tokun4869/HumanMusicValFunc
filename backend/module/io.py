# ===== ===== ===== =====
# io Module
# ===== ===== ===== =====

from pickle import LIST
import numpy as np
import librosa
import glob
import re
import torch
import csv
from module.model import Model
from module.exception import InvalidArgumentException
from module.const import *


def __atoi(text: str) -> (int | str):
  """
  文字列を整数型に変換する
  
  Parameters
  ----------
  text : str
    対象の文字列
  
  Returns
  ----------
  transformed_text : int | str
    変換後の数字もしくは文字列
  """

  return int(text) if text.isdigit() else text


def __natural_keys(text: str) -> "list[int | str]":
  """
  文字列に含まれる数字を整数型に変換する
  
  Parameters
  ----------
  text : str
    対象の文字列
  
  Returns
  ----------
  transformded_text : list[int | str]
    変換後の数字もしくは文字列の配列
  """

  return [__atoi(c) for c in re.split(r"(\d+)", text)]


def get_file_name_list(dir: str, ext: str = MUSIC_EXT) -> "list[str]":
  """
  ディレクトリ内のファイルを検索する
  
  Parameters
  ----------
  dir : str
    対象のディレクトリ
  ext : str
    対象の拡張子
  
  Returns
  ----------
  searched_file_name : list[str]
    該当したファイルパスの配列
  """

  search_text = f"{dir}/*{ext}"
  return sorted(glob.glob(search_text), key=__natural_keys)


def load_sound(path: str) -> "np.ndarray[np.float32]":
  """
  音声ファイルをnumpy.ndarray形式の音声波形データとして読み込む
  
  Parameters
  ----------
  path : str
    対象の音声ファイルのパス
  
  Returns
  ----------
  sound : numpy.ndarray[numpy.float32]
    音声波形の配列
  """

  sound, _ = librosa.load(path, sr=SAMPLE_RATE)
  return sound

def load_sound_list(path_list: "list[str]") -> "list[np.ndarray[np.float32]]":
  """
  配列で与えられた音声ファイルをnumpy.ndarray形式の音声波形データとして読み込む
  
  Parameters
  ----------
  path_list : list[str]
    読み込むファイルパスの配列
  
  Returns
  ----------
  sound_list : list[np.ndarray[np.float32]]
    ファイルパスの配列
  """

  sound_list = []
  for path in path_list:
    sound_list.append(load_sound(path))
  return sound_list


def get_new_file_path(dir: str, base: str, ext: str) -> str:
  """
  既存ファイル名から連番のファイル名を求める
  
  Parameters
  ----------
  dir : str
    対象のディレクトリ
  base : str
    対象の連番部分を除くファイル名
  ext : str
    対象の拡張子
  
  Returns
  ----------
  path : str
    新しい連番のファイルパス
  """

  file_name_list = glob.glob(f"{dir}/{base}_*{ext}")
  max_index = -1
  for file_name in file_name_list:
    index = int(file_name[len(f"{dir}/{base}_") : -len(ext)])
    if max_index < index:
      max_index = index
  path = f"{dir}/{base}_{max_index+1}{ext}"
  return path


def load_model(path: str, extractor: str = EXTRACTOR_TYPE, head: str = HEAD_TYPE, is_eval: bool = True, device: torch.device=torch.device("cpu")) -> Model:
  """
  保存されたnn.Moduleモデルを読み込む
  
  Parameters
  ----------
  path : str
    対象のファイルパス
  extractor : str
    対象の抽出モデル
  head : str
    対象の回帰モデル
  is_eval : bool
    読み込み時のモデルが評価モードか
  
  Returns
  ----------
  model : ReprMPL | ReprLinearRegresssion | SpecCNN | FeatCNN
    読み込んだモデル
  """

  model = Model(extractor, head, device)
  model.load_state_dict(torch.load(path))
  if is_eval:
    model.eval()
  else:
    model.train()
  return model


def get_dataset_path(extractor_type: str = EXTRACTOR_TYPE) -> str:
  """
  設定からデータセットのパスを取得する
  
  Returns
  ----------
  path : str
    取得したパス
  """
  if extractor_type == EXTRACTOR_REPR:
    return f"{DATASET_ROOT}/{DATASET_TYPE}/{MODE}/{REPR_DATASET_BASE}{LIST_EXT}"
  if extractor_type == EXTRACTOR_SPEC:
    return f"{DATASET_ROOT}/{DATASET_TYPE}/{MODE}/{SPEC_DATASET_BASE}{LIST_EXT}"
  if extractor_type == EXTRACTOR_FEAT:
    return f"{DATASET_ROOT}/{DATASET_TYPE}/{MODE}/{FEAT_DATASET_BASE}{LIST_EXT}"
  else:
    raise InvalidArgumentException(extractor_type)


def load_dataset(dataset_path: str, answer_list: "list[int]", file_name_list: "list[str]" = get_file_name_list(f"{MUSIC_ROOT}/{DATASET_TYPE}/{MODE}_{LISTEN_KEY}")) -> "tuple[list, list[int]]":
  """
  音声ファイルセットと回答データからデータセット用の入力と正解の組の配列を作る
  
  Parameters
  ----------
  dataset_path : str
    対象のデータセット一覧のファイルパス
  answer_list : list[int]
    回答データ
  file_name_list : list[str]
    データセットのもとになった音声ファイルパスの配列
  
  Returns
  ----------
  input_list : list[numpy.ndarray]
    入力データの配列
  target_list : list[numpy.ndarray]
    正解データの配列
  """

  input_list = []
  target_list = []
  with open(dataset_path, "r") as f:
    reader = csv.reader(f)
    for row in reader:
      input_list.append(np.load(row[0]))
      target_list.append(answer_list[file_name_list.index(row[1])])
  return input_list, target_list


def save_user_data(dataset: str, name: str, train_list: list[int], test_list: list[int], retest_list: list[int]) -> None:
  """
  回答データを保存する
  
  Parameters
  ----------
  name : str
    回答者の名前
  train_list : list[int]
    学習データに対する回答データ
  test_list : list[int]
    テストデータに対する回答データ
  retest_list : list[int]
    再学習用テストデータに対する回答データ
  """
  
  file_name = f"{name}{LIST_EXT}"
  with open(f"{USER_ROOT}/{dataset}/{file_name}", "w") as f:
    writer = csv.writer(f)
    writer.writerow(train_list)
    writer.writerow(test_list)
    writer.writerow(retest_list)


def load_user_data(file_name: str) -> list[int]:
  """
  指定した回答データを読み込む

  Parameters
  ----------
  file_path: str
    回答データのパス
  
  Returns
  ----------
  target_list : list[int]
    正解データの配列
  """
  with open(file_name, "r") as f:
    reader = csv.reader(f)
    l = [row for row in reader]
    if MODE == MODE_TEST:
      target_list = [int(data) for data in l[1]]
    if MODE == MODE_RETEST:
      target_list = [int(data) for data in l[2]]
    else:
      target_list = [int(data) for data in l[0]]
  
  return target_list


def get_label_from_user_data(file_path: str) -> str:
  """
  指定した回答データからユーザ名とデータセット名からなるラベルを取り出す

  Parameters
  ----------
  file_path: str
    回答データのパス
  
  Returns
  ----------
  label : str
    ユーザ名とデータセット名からなるラベル
  """
  file_name = file_path.split("/")[-1]
  user_name = file_name.split(".")[0]
  dataset_name = file_path.split("/")[-2]
  label = f"{dataset_name}_{EXTRACTOR_TYPE}_{user_name}"

  return label


def load_all_user_data() -> tuple[list[list[int]], list[str]]:
  """
  全回答データを読み込む
  
  Returns
  ----------
  answer_list_list : list[list[int]]
    全員の正解データの配列
  label_list : list[list[int]]
    全員のラベルの配列
  """
  file_path_list = get_file_name_list(f"{USER_ROOT}/{DATASET_TYPE}", ext=LIST_EXT)
  label_list = [get_label_from_user_data(file_path) for file_path in file_path_list]
  answer_list_list = [load_user_data(file_path) for file_path in file_path_list]
  
  return answer_list_list, label_list