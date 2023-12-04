# ===== ===== ===== =====
# Const Define Module
# ===== ===== ===== =====


DATASET_TYPE = "MAESTRO"
MODEL_TYPE = "ReprMLP"
MODE = "train"

SAMPLE_RATE = 22050
LENGTH = 30

LEARNING_RATE = 0.0001
NUM_EPOCHS = 300
BATCH_SIZE = 3
SEED = 42

MUSIC_ROOT = "data/music"
DATASET_ROOT = "data/list"
GRAPH_ROOT = "graph"
LOSS_ROOT = "loss"
MODEL_ROOT = "model"
RESULT_ROOT = "result"
USER_ROOT = "user"

STATUS_BEFORE = "before"
STATUS_INPROGRESS = "inprogress"
STATUS_FINISH = "finish"
STATUS_ERROR = "error"

DATASET_MAESTRO = "MAESTRO"
DATASET_MUSICNET = "MusicNet"
DATASET_MTG = "MTG-Jamendo"

MODEL_MLP = "ReplMLP"
MODEL_LR = "ReplLR"
MODEL_SPEC = "SpecCNN"
MODEL_FEAT = "FeatCNN"

MODE_TRAIN = "train"
MODE_TEST = "test"
MODE_RETEST = "re-test"

SPEC_DATASET_BASE = "spec_dataset"
FEAT_DATASET_BASE = "feat_dataset"
REPR_DATASET_BASE = "repr_dataset"

LIST_EXT = ".csv"
GRAPH_EXT = ".svg"
MODEL_EXT = ".pth"
MUSIC_EXT = ".mp3"

LISTEN_KEY = "listen"
INPUT_KEY = "input"