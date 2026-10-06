"""
State-of-the-Art Cross-Validation Training and Stacking Pipeline.
Trains 5-Fold Stratified on Genuine Speech Samples (N=732):
1. Support Vector Regressor (SVR - RBF kernel, C=3.0, epsilon=0.08) on Standardized Features
2. XGBoost Regressor with Subsampling and L1/L2 Regularization
3. LightGBM Regressor with Regularized Leaf Splits
4. HistGradientBoosting Regressor for Monotonicity and Robustness
5. BayesianRidge Regressor for Regularized Linear Projection

Key Innovations:
- Drop duration artifact features ('audio_duration', 'whisper_speech_duration') that induce test-time shift.
- Static noise files (ZCR > 0.35) are separated deterministically to 0.0.
- Regression models trained strictly on genuine speech data.
- SLSQP constrained blend optimization directly minimizing Out-Of-Fold RMSE.
- Centered dynamic range variance calibration aligning test predictions strictly to the true speech distribution (Mean 3.48).
- Computes and logs compulsory Training Data RMSE and OOF evaluation metrics.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb

from src.config import (
    FEATURES_TRAIN_FILE,
    MODELS_DIR,
    RANDOM_SEED,
    N_SPLITS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(root_mean_squared_error(y_true, y_pred))
    p_corr, _ = pearsonr(y_true, y_pred)
    s_corr, _ = spearmanr(y_true, y_pred)
    return {
        "rmse": rmse,
        "pearson": float(p_corr),
        "spearman": float(s_corr),
    }


def train_models():
    logger.info(f"Loading upgraded training features from {FEATURES_TRAIN_FILE}...")
    df = pd.read_parquet(FEATURES_TRAIN_FILE)

    target_col = "label"
    # Drop duration artifact features that cause test-time distribution shifts
    drop_cols = ["filename", target_col, "audio_duration", "whisper_speech_duration"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    logger.info(f"Using {len(feature_cols)} clean features across {len(df)} total samples.")

    # 1. Deterministic Acoustic Noise Filtering
    noise_mask = (df["zcr_mean"] > 0.35).values
    logger.info(f"Detected {np.sum(noise_mask)} static noise samples (ZCR > 0.35). Assigned label 0.0.")

    # 2. Partition Genuine Speech Samples for High-Precision Modeling
    df_speech = df[~noise_mask].copy().reset_index(drop=True)
    X_speech = df_speech[feature_cols].copy().fillna(0.0)
    y_speech = df_speech[target_col].values
    logger.info(f"Training core regression ensemble strictly on {len(df_speech)} speech samples.")
    logger.info(f"Speech Ground Truth Stats: Mean={np.mean(y_speech):.4f}, Std={np.std(y_speech):.4f}, Min={np.min(y_speech):.1f}, Max={np.max(y_speech):.1f}")

    # 3. Stratified 5-Fold Partitioning
    bins = [0.9, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.1]
    y_bins = pd.cut(y_speech, bins=bins, labels=False)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    oof_svr = np.zeros(len(df_speech))
    oof_xgb = np.zeros(len(df_speech))
    oof_lgb = np.zeros(len(df_speech))
    oof_hgb = np.zeros(len(df_speech))
    oof_bridge = np.zeros(len(df_speech))

    models_svr = []
    models_xgb = []
    models_lgb = []
    models_hgb = []
    models_bridge = []
    scalers = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_speech, y_bins)):
        X_tr, y_tr = X_speech.iloc[train_idx], y_speech[train_idx]
        X_va, y_va = X_speech.iloc[val_idx], y_speech[val_idx]

        # Standard Scaler for kernel and linear models (fit on train fold only)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        # 1. Support Vector Regressor (SVR - RBF kernel)
        m_svr = SVR(C=3.0, epsilon=0.08, gamma="scale")
        m_svr.fit(X_tr_s, y_tr)
        oof_svr[val_idx] = m_svr.predict(X_va_s)
        models_svr.append(m_svr)

        # 2. XGBoost Regressor
        m_xgb = xgb.XGBRegressor(
            n_estimators=700,
            learning_rate=0.02,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.3,
            reg_lambda=2.0,
            random_state=RANDOM_SEED + fold,
            n_jobs=-1,
            early_stopping_rounds=60,
        )
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[val_idx] = m_xgb.predict(X_va)
        models_xgb.append(m_xgb)

        # 3. LightGBM Regressor
        m_lgb = lgb.LGBMRegressor(
            n_estimators=700,
            learning_rate=0.02,
            num_leaves=24,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.3,
            reg_lambda=2.0,
            min_child_samples=12,
            random_state=RANDOM_SEED + fold,
            verbose=-1,
            n_jobs=-1,
        )
        m_lgb.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
        )
        oof_lgb[val_idx] = m_lgb.predict(X_va)
        models_lgb.append(m_lgb)

        # 4. HistGradientBoosting Regressor
        m_hgb = HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.02,
            max_leaf_nodes=24,
            l2_regularization=1.5,
            min_samples_leaf=12,
            random_state=RANDOM_SEED + fold,
        )
        m_hgb.fit(X_tr, y_tr)
        oof_hgb[val_idx] = m_hgb.predict(X_va)
        models_hgb.append(m_hgb)

        # 5. BayesianRidge Regressor
        m_bridge = BayesianRidge(alpha_1=1e-4, alpha_2=1e-4, lambda_1=1e-4, lambda_2=1e-4)
        m_bridge.fit(X_tr_s, y_tr)
        oof_bridge[val_idx] = m_bridge.predict(X_va_s)
        models_bridge.append(m_bridge)

        scalers.append(scaler)

        logger.info(
            f"Fold {fold+1} | "
            f"SVR: {root_mean_squared_error(y_va, oof_svr[val_idx]):.4f} | "
            f"XGB: {root_mean_squared_error(y_va, oof_xgb[val_idx]):.4f} | "
            f"LGB: {root_mean_squared_error(y_va, oof_lgb[val_idx]):.4f} | "
            f"HGB: {root_mean_squared_error(y_va, oof_hgb[val_idx]):.4f} | "
            f"BRidge: {root_mean_squared_error(y_va, oof_bridge[val_idx]):.4f}"
        )

    logger.info("=== Single Model Out-Of-Fold Performance on Speech ===")
    logger.info(f"SVR (RBF)   -> {compute_metrics(y_speech, np.clip(oof_svr, 1.0, 5.0))}")
    logger.info(f"XGBoost     -> {compute_metrics(y_speech, np.clip(oof_xgb, 1.0, 5.0))}")
    logger.info(f"LightGBM    -> {compute_metrics(y_speech, np.clip(oof_lgb, 1.0, 5.0))}")
    logger.info(f"HistGB      -> {compute_metrics(y_speech, np.clip(oof_hgb, 1.0, 5.0))}")
    logger.info(f"BayesianR   -> {compute_metrics(y_speech, np.clip(oof_bridge, 1.0, 5.0))}")

    # 4. Optimize Ensemble Blend Weights to DIRECTLY minimize Speech RMSE
    all_oofs = [oof_svr, oof_xgb, oof_lgb, oof_hgb, oof_bridge]

    def blend_loss(weights):
        pred = sum(w * oof for w, oof in zip(weights, all_oofs))
        pred = np.clip(pred, 1.0, 5.0)
        return root_mean_squared_error(y_speech, pred)

    init_weights = [0.45, 0.20, 0.10, 0.15, 0.10]
    opt_res = minimize(
        blend_loss,
        init_weights,
        bounds=[(0, 1)] * 5,
        constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
    )
    best_weights = opt_res.x
    logger.info(
        f"Optimal Weights: SVR={best_weights[0]:.3f}, XGB={best_weights[1]:.3f}, "
        f"LGB={best_weights[2]:.3f}, HGB={best_weights[3]:.3f}, BRidge={best_weights[4]:.3f}"
    )

    oof_blend = sum(w * oof for w, oof in zip(best_weights, all_oofs))

    # 5. Centered Dynamic Range Variance & Mean Calibration on Speech
    target_speech_mean = float(np.mean(y_speech))
    oof_blend_mean = float(np.mean(oof_blend))

    def scale_loss(s):
        cal = np.clip(target_speech_mean + s[0] * (oof_blend - oof_blend_mean), 1.5, 5.0)
        return root_mean_squared_error(y_speech, cal)

    opt_scale = minimize(scale_loss, [1.15], bounds=[(0.8, 1.6)])
    cal_scale = float(opt_scale.x[0])
    logger.info(f"Learned Centered Calibration: target_mean={target_speech_mean:.4f}, scale={cal_scale:.4f}")

    oof_speech_cal = np.clip(target_speech_mean + cal_scale * (oof_blend - oof_blend_mean), 1.5, 5.0)
    speech_metrics = compute_metrics(y_speech, oof_speech_cal)

    logger.info("=====================================================")
    logger.info("=== SPEECH OUT-OF-FOLD PERFORMANCE ===")
    logger.info(f"Speech OOF RMSE:         {speech_metrics['rmse']:.4f}")
    logger.info(f"Speech OOF Pearson r:    {speech_metrics['pearson']:.4f}")
    logger.info(f"Speech OOF Spearman rho: {speech_metrics['spearman']:.4f}")
    logger.info(f"Predicted Speech Mean:   {np.mean(oof_speech_cal):.4f} (Target: {target_speech_mean:.4f})")
    logger.info(f"Predicted Speech Std:    {np.std(oof_speech_cal):.4f} (Target: {np.std(y_speech):.4f})")
    logger.info("=====================================================")

    # 6. Full Dataset OOF Metrics (combining speech and noise)
    y_full = df[target_col].values
    oof_full = np.zeros(len(df))
    oof_full[~noise_mask] = oof_speech_cal
    oof_full[noise_mask] = 0.0

    oof_full_metrics = compute_metrics(y_full, oof_full)
    logger.info("=== FULL DATASET OUT-OF-FOLD METRICS ===")
    logger.info(f"Full OOF RMSE:         {oof_full_metrics['rmse']:.4f}")
    logger.info(f"Full OOF Pearson r:    {oof_full_metrics['pearson']:.4f}")
    logger.info(f"Full OOF Spearman rho: {oof_full_metrics['spearman']:.4f}")
    logger.info("=====================================================")

    # 7. COMPULSORY Training Data RMSE (Full train evaluated across all fold models)
    preds_full_svr = np.mean([m.predict(s.transform(X_speech)) for m, s in zip(models_svr, scalers)], axis=0)
    preds_full_xgb = np.mean([m.predict(X_speech) for m in models_xgb], axis=0)
    preds_full_lgb = np.mean([m.predict(X_speech) for m in models_lgb], axis=0)
    preds_full_hgb = np.mean([m.predict(X_speech) for m in models_hgb], axis=0)
    preds_full_bridge = np.mean([m.predict(s.transform(X_speech)) for m, s in zip(models_bridge, scalers)], axis=0)

    train_blend_speech = (
        best_weights[0] * preds_full_svr
        + best_weights[1] * preds_full_xgb
        + best_weights[2] * preds_full_lgb
        + best_weights[3] * preds_full_hgb
        + best_weights[4] * preds_full_bridge
    )
    train_cal_speech = np.clip(target_speech_mean + cal_scale * (train_blend_speech - np.mean(train_blend_speech)), 1.5, 5.0)

    train_full_preds = np.zeros(len(df))
    train_full_preds[~noise_mask] = train_cal_speech
    train_full_preds[noise_mask] = 0.0

    train_metrics = compute_metrics(y_full, train_full_preds)
    logger.info("=====================================================")
    logger.info(f"COMPULSORY TRAINING DATA RMSE: {train_metrics['rmse']:.4f}")
    logger.info(f"Training Data Pearson r:       {train_metrics['pearson']:.4f}")
    logger.info(f"Training Data Spearman rho:    {train_metrics['spearman']:.4f}")
    logger.info("=====================================================")

    # 8. Save models and evaluation metadata
    save_payload = {
        "models_svr": models_svr,
        "models_xgb": models_xgb,
        "models_lgb": models_lgb,
        "models_hgb": models_hgb,
        "models_bridge": models_bridge,
        "scalers": scalers,
        "feature_cols": feature_cols,
        "best_weights": best_weights.tolist(),
        "target_speech_mean": float(target_speech_mean),
        "cal_scale": float(cal_scale),
        "speech_metrics": speech_metrics,
        "oof_full_metrics": oof_full_metrics,
        "train_metrics": train_metrics,
    }

    model_path = MODELS_DIR / "ensemble_models.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(save_payload, f)
    logger.info(f"Saved trained ensemble to {model_path}")

    summary_path = MODELS_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "training_metrics": train_metrics,
                "speech_oof_metrics": speech_metrics,
                "full_oof_metrics": oof_full_metrics,
                "calibration": {"target_speech_mean": float(target_speech_mean), "scale": float(cal_scale)},
                "weights": {
                    "svr": float(best_weights[0]),
                    "xgb": float(best_weights[1]),
                    "lgb": float(best_weights[2]),
                    "hgb": float(best_weights[3]),
                    "bridge": float(best_weights[4]),
                },
            },
            f,
            indent=2,
        )
    return save_payload


def main():
    train_models()


if __name__ == "__main__":
    main()
