"""
Upgraded Inference and Submission Generation Module.
Loads the 5-model ensemble (SVR, XGBoost, LightGBM, HistGB, BayesianRidge),
applies optimal blend weights and centered target distribution variance calibration,
generates test predictions strictly formatted as sample_submission.csv,
and executes rigorous validation checks.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.config import (
    FEATURES_TEST_FILE,
    TEST_CSV,
    SAMPLE_SUBMISSION_CSV,
    MODELS_DIR,
    SUBMISSIONS_DIR,
    BASE_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_predictions():
    model_path = MODELS_DIR / "ensemble_models.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained models not found at {model_path}. Run src/train.py first.")

    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    models_svr = payload["models_svr"]
    models_xgb = payload["models_xgb"]
    models_lgb = payload["models_lgb"]
    models_hgb = payload["models_hgb"]
    models_bridge = payload["models_bridge"]
    scalers = payload["scalers"]
    feature_cols = payload["feature_cols"]
    best_weights = payload["best_weights"]
    target_speech_mean = payload.get("target_speech_mean", 3.4809)
    cal_scale = payload.get("cal_scale", 1.15)

    logger.info(f"Loading test features from {FEATURES_TEST_FILE}...")
    df_test = pd.read_parquet(FEATURES_TEST_FILE)

    X_test = df_test[feature_cols].copy().fillna(0.0)

    # 1. Fold-average predictions for each model family
    preds_svr = np.mean([m.predict(s.transform(X_test)) for m, s in zip(models_svr, scalers)], axis=0)
    preds_xgb = np.mean([m.predict(X_test) for m in models_xgb], axis=0)
    preds_lgb = np.mean([m.predict(X_test) for m in models_lgb], axis=0)
    preds_hgb = np.mean([m.predict(X_test) for m in models_hgb], axis=0)
    preds_bridge = np.mean([m.predict(s.transform(X_test)) for m, s in zip(models_bridge, scalers)], axis=0)

    # 2. Optimal 5-Model Weighted Blend
    raw_blend = (
        best_weights[0] * preds_svr
        + best_weights[1] * preds_xgb
        + best_weights[2] * preds_lgb
        + best_weights[3] * preds_hgb
        + best_weights[4] * preds_bridge
    )

    # 3. Centered Target Distribution Variance Calibration (matching empirical speech distribution)
    test_raw_mean = np.mean(raw_blend)
    logger.info(f"Test Raw Blend Mean: {test_raw_mean:.4f}, Scaling to Target Mean: {target_speech_mean:.4f} with scale: {cal_scale:.4f}")
    calibrated_preds = target_speech_mean + cal_scale * (raw_blend - test_raw_mean)

    # 4. Deterministic acoustic noise gate
    noise_mask = (df_test["zcr_mean"] > 0.35).values
    calibrated_preds[noise_mask] = 0.0

    # 5. Strict clamping to [0.0, 5.0]
    final_preds = np.clip(calibrated_preds, 0.0, 5.0)

    df_test["predicted_label"] = np.round(final_preds, 4)

    # 1. Primary Submission: For the 216 files evaluated in test.csv
    df_submission = pd.DataFrame(
        {
            "filename": df_test["filename"],
            "label": df_test["predicted_label"],
        }
    )

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission_path_dir = SUBMISSIONS_DIR / "submission.csv"
    submission_path_root = BASE_DIR / "submission.csv"

    df_submission.to_csv(submission_path_dir, index=False)
    df_submission.to_csv(submission_path_root, index=False)
    logger.info(f"Saved primary submission to {submission_path_root} ({len(df_submission)} rows)")

    # 2. Secondary Compatible Submission: Formatted matching sample_submission.csv rows
    if SAMPLE_SUBMISSION_CSV.exists():
        df_sample = pd.read_csv(SAMPLE_SUBMISSION_CSV)
        pred_map = dict(zip(df_test["filename"], df_test["predicted_label"]))
        fallback_val = float(np.round(np.mean(final_preds), 4))

        matched_preds = []
        for fn in df_sample["filename"]:
            matched_preds.append(pred_map.get(fn, fallback_val))

        df_sample_filled = pd.DataFrame(
            {
                "filename": df_sample["filename"],
                "label": matched_preds,
            }
        )
        sample_filled_path = SUBMISSIONS_DIR / "sample_submission_filled.csv"
        df_sample_filled.to_csv(sample_filled_path, index=False)
        logger.info(f"Saved sample-submission-matched file to {sample_filled_path} ({len(df_sample_filled)} rows)")

    # 3. Validation checks
    validate_submission(submission_path_root)


def validate_submission(submission_path: Path):
    logger.info(f"Running validation checks on {submission_path}...")
    df = pd.read_csv(submission_path)

    assert "filename" in df.columns and "label" in df.columns, "Columns must be ['filename', 'label']"
    assert len(df.columns) == 2, f"Expected exactly 2 columns, got {len(df.columns)}"
    assert not df["label"].isnull().any(), "Found NaN values in label column!"
    assert not np.isinf(df["label"]).any(), "Found infinite values in label column!"
    assert (df["label"] >= 0.0).all() and (df["label"] <= 5.0).all(), "Predictions out of range [0.0, 5.0]!"

    logger.info("Validation passed successfully!")
    logger.info(f"Summary Statistics:\n{df['label'].describe()}")
    logger.info(f"Percentiles:\n{df['label'].quantile([0.05, 0.25, 0.5, 0.75, 0.95])}")
    logger.info(f"First 10 predictions:\n{df.head(10)}")


def main():
    generate_predictions()


if __name__ == "__main__":
    main()
