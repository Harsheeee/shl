import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
TRAIN_CSV = DATASET_DIR / "train.csv"
TEST_CSV = DATASET_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATASET_DIR / "sample_submission.csv"

# Artifacts & Outputs
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

TRANSCRIPTS_TRAIN_FILE = ARTIFACTS_DIR / "transcripts_train.json"
TRANSCRIPTS_TEST_FILE = ARTIFACTS_DIR / "transcripts_test.json"
FEATURES_TRAIN_FILE = ARTIFACTS_DIR / "features_train.parquet"
FEATURES_TEST_FILE = ARTIFACTS_DIR / "features_test.parquet"

# Models & Hardware
WHISPER_MODEL_NAME = "openai/whisper-tiny.en"
TEXT_EMBEDDING_MODEL = "distilbert-base-uncased"
AUDIO_SAMPLE_RATE = 16000
RANDOM_SEED = 42
N_SPLITS = 5

for d in [ARTIFACTS_DIR, MODELS_DIR, SUBMISSIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
