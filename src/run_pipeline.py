"""
Master Pipeline Runner for Spoken Grammar Scoring Engine.
Orchestrates:
1. Feature extraction
2. Cross-validation model training & evaluation
3. Inference and final submission file generation
"""

import logging
import sys
from pathlib import Path

from src.config import (
    TRANSCRIPTS_TRAIN_FILE,
    TRANSCRIPTS_TEST_FILE,
    FEATURES_TRAIN_FILE,
    FEATURES_TEST_FILE,
    MODELS_DIR,
    SUBMISSIONS_DIR,
    BASE_DIR,
)
from src.extract_features import main as extract_main
from src.train import main as train_main
from src.predict import main as predict_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_full_pipeline():
    logger.info("==================================================")
    logger.info("   SPOKEN GRAMMAR SCORING ENGINE - PIPELINE")
    logger.info("==================================================")

    # 1. Check Transcripts
    if not TRANSCRIPTS_TRAIN_FILE.exists() or not TRANSCRIPTS_TEST_FILE.exists():
        logger.error("Transcripts not found! Please run src/transcribe.py first.")
        sys.exit(1)

    # 2. Multimodal Feature Extraction
    logger.info("\n--- STEP 1: MULTIMODAL FEATURE EXTRACTION ---")
    extract_main()

    # 3. Model Training & Validation
    logger.info("\n--- STEP 2: CROSS-VALIDATION & MODEL ENSEMBLING ---")
    train_main()

    # 4. Inference & Submission Generation
    logger.info("\n--- STEP 3: INFERENCE & SUBMISSION GENERATION ---")
    predict_main()

    logger.info("\n==================================================")
    logger.info("   PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info(f"   Final submission: {BASE_DIR / 'submission.csv'}")
    logger.info("==================================================")


if __name__ == "__main__":
    run_full_pipeline()
