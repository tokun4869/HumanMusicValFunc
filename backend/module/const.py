# ===== ===== ===== =====
# Const Define Module
# ===== ===== ===== =====


DATASET_TYPE = "MAESTRO"
EXTRACTOR_TYPE = "REPR"
HEAD_TYPE = "MLP"
MODE = "train"

SAMPLE_RATE = 22050
LENGTH = 30
N_FFT = 2048
WIN_LENGTH = 2048
HOP_LENGTH = 512

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
TIME_ROOT = "time"

STATUS_BEFORE = "before"
STATUS_INPROGRESS = "inprogress"
STATUS_FINISH = "finish"
STATUS_ERROR = "error"

DATASET_MAESTRO = "MAESTRO"
DATASET_MUSICNET = "MusicNet"
DATASET_MTG = "MTG-Jamendo"

EXTRACTOR_REPR = "REPR"
EXTRACTOR_SPEC = "SPEC"
EXTRACTOR_FEAT = "FEAT"

HEAD_MLP = "MLP"
HEAD_LR = "LR"

MODE_TRAIN = "train"
MODE_TEST = "test"
MODE_RETEST = "re-test"

WAVE_DATASET_BASE = "wave_dataset"
SPEC_DATASET_BASE = "spec_dataset"
FEAT_DATASET_BASE = "feat_dataset"
REPR_DATASET_BASE = "repr_dataset"

LIST_EXT = ".csv"
GRAPH_EXT = ".svg"
MODEL_EXT = ".pth"
MUSIC_EXT = ".mp3"

LISTEN_KEY = "listen"
INPUT_KEY = "input"

C1_HZ = 33.661
B7_HZ = 4066.841