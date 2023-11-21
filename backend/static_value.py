DATASET_TYPE = "MAESTRO"
IS_NORM = True
IS_STD = False
IS_FEAT = True

SAMPLE_RATE = 22050
LENGTH = 30
MUSIC_ROOT = "data/music/"
DATASET_ROOT = "data/list/"
MODEL_ROOT = "model/"
GRAPH_ROOT = "graph/"
FEATURE_ROOT = "feature/"
RESULT_ROOT = "result/"
USER_ROOT = "user/"
TRAIN_BASE = "train"
TARGET_BASE = "target"
MODEL_BASE = "model"
STATUS_BEFORE = "before"
STATUS_INPROGRESS = "inprogress"
STATUS_FINISH = "finish"
STATUS_ERROR = "error"
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
BATCH_SIZE = 3
SEED = 42

if DATASET_TYPE == "OURS":
    TRAIN_LISTEN_DIR = "OURS/train_listen/"
    TRAIN_INPUT_DIR = "OURS/train_input/"
    TEST_LISTEN_DIR = "OURS/test_listen/"
    TEST_INPUT_DIR = "OURS/test_input/"
    WAVE_DATASET_BASE = "train_wave_dataset_0.csv"
    SPEC_DATASET_BASE = "train_spec_dataset_0.csv"
    FEAT_DATASET_BASE = "train_feat_dataset_0.csv"
elif DATASET_TYPE == "MAESTRO":
    TRAIN_LISTEN_DIR = "MAESTRO/train_listen/"
    TRAIN_INPUT_DIR = "MAESTRO/train_input/"
    TEST_LISTEN_DIR = "MAESTRO/test_listen/"
    TEST_INPUT_DIR = "MAESTRO/test_input/"
    WAVE_DATASET_BASE = "train_wave_dataset_0.csv"
    SPEC_DATASET_BASE = "train_spec_dataset_0.csv"
    FEAT_DATASET_BASE = "train_feat_dataset_0.csv"