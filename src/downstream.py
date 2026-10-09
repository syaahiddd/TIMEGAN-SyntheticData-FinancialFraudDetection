"""
Downstream Utility Benchmarking for Financial Fraud Detection:
Implements Train-on-Real, Test-on-Real (TRTR) and 
Train-on-Synthetic, Test-on-Real (TSTR) pipelines using XGBoost.
Computes PR-AUC, ROC-AUC, F1-Score, and Utility Retention Ratios.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    classification_report
)
from typing import Dict, Tuple, Optional


class DownstreamBenchmark:
    """
    Evaluates downstream fraud classification utility of synthetic time-series.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _flatten_or_pool_sequences(
        self, sequences: np.ndarray, method: str = "flatten"
    ) -> np.ndarray:
        """
        Converts 3D sequence tensors (N, T, D) into 2D feature matrices (N, Features)
        for tabular gradient boosting classifiers.
        
        Args:
            method: 'flatten' (preserves step positions) or 
                    'summary' (calculates mean, std, min, max across time steps)
        """
        n_samples, seq_len, n_features = sequences.shape

        if method == "flatten":
            return sequences.reshape(n_samples, seq_len * n_features)
        
        elif method == "summary":
            mean_feat = np.mean(sequences, axis=1)
            std_feat = np.std(sequences, axis=1)
            min_feat = np.min(sequences, axis=1)
            max_feat = np.max(sequences, axis=1)
            return np.hstack([mean_feat, std_feat, min_feat, max_feat])
        
        else:
            raise ValueError(f"Unknown pooling method: {method}")

    def _train_and_evaluate_xgb(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        scale_pos_weight: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Fits XGBoost and returns ROC-AUC, PR-AUC, and F1.
        """
        # Automatically account for severe fraud class imbalance if not set
        if scale_pos_weight is None:
            n_neg = np.sum(y_train == 0)
            n_pos = np.sum(y_train == 1)
            scale_pos_weight = float(n_neg / max(1, n_pos))

        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            n_jobs=-1
        )

        model.fit(x_train, y_train)

        # Probabilities for positive class (Fraud)
        y_probs = model.predict_proba(x_test)[:, 1]
        y_preds = (y_probs >= 0.5).astype(int)

        roc_auc = float(roc_auc_score(y_test, y_probs))
        pr_auc = float(average_precision_score(y_test, y_probs))
        f1 = float(f1_score(y_test, y_preds, zero_division=0))

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1_score": f1
        }

    def evaluate_tstr_vs_trtr(
        self,
        real_train_x: np.ndarray,
        real_train_y: np.ndarray,
        synth_train_x: np.ndarray,
        synth_train_y: np.ndarray,
        real_test_x: np.ndarray,
        real_test_y: np.ndarray,
        repr_method: str = "flatten"
    ) -> pd.DataFrame:
        """
        Runs both TRTR and TSTR pipelines and compiles comparison metrics.
        """
        # Transform 3D tensors into 2D tabular features
        x_tr_real = self._flatten_or_pool_sequences(real_train_x, method=repr_method)
        x_tr_synth = self._flatten_or_pool_sequences(synth_train_x, method=repr_method)
        x_te_real = self._flatten_or_pool_sequences(real_test_x, method=repr_method)

        print("\n--- Training TRTR Baseline (Real -> Real) ---")
        trtr_metrics = self._train_and_evaluate_xgb(
            x_tr_real, real_train_y, x_te_real, real_test_y
        )

        print("--- Training TSTR Benchmark (Synthetic -> Real) ---")
        tstr_metrics = self._train_and_evaluate_xgb(
            x_tr_synth, synth_train_y, x_te_real, real_test_y
        )

        # Calculate retention percentage: (TSTR / TRTR) * 100
        comparison = {
            "Metric": ["ROC-AUC", "PR-AUC", "F1-Score"],
            "TRTR (Real)": [
                trtr_metrics["roc_auc"],
                trtr_metrics["pr_auc"],
                trtr_metrics["f1_score"]
            ],
            "TSTR (Synthetic)": [
                tstr_metrics["roc_auc"],
                tstr_metrics["pr_auc"],
                tstr_metrics["f1_score"]
            ],
            "Utility Retention (%)": [
                (tstr_metrics["roc_auc"] / (trtr_metrics["roc_auc"] + 1e-8)) * 100,
                (tstr_metrics["pr_auc"] / (trtr_metrics["pr_auc"] + 1e-8)) * 100,
                (tstr_metrics["f1_score"] / (trtr_metrics["f1_score"] + 1e-8)) * 100
            ]
        }

        df_results = pd.DataFrame(comparison)
        return df_results


# -------------------------------------------------------------------------
# Smoke Test / Module Verification
# -------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing downstream TSTR vs. TRTR benchmark module...")

    np.random.seed(42)
    N_TRAIN = 600
    N_TEST = 200
    SEQ_LEN = 24
    N_FEATS = 5

    # Generate synthetic mock temporal sequences
    real_train_seq = np.random.uniform(0.0, 1.0, size=(N_TRAIN, SEQ_LEN, N_FEATS)).astype(np.float32)
    synth_train_seq = real_train_seq + np.random.normal(0, 0.05, size=(N_TRAIN, SEQ_LEN, N_FEATS)).astype(np.float32)
    real_test_seq = np.random.uniform(0.0, 1.0, size=(N_TEST, SEQ_LEN, N_FEATS)).astype(np.float32)

    # Imbalanced fraud targets (5% fraud)
    y_real_tr = (np.random.rand(N_TRAIN) > 0.95).astype(int)
    y_synth_tr = y_real_tr.copy()
    y_real_te = (np.random.rand(N_TEST) > 0.95).astype(int)

    benchmark = DownstreamBenchmark(random_state=42)
    results_df = benchmark.evaluate_tstr_vs_trtr(
        real_train_seq, y_real_tr,
        synth_train_seq, y_synth_tr,
        real_test_seq, y_real_te,
        repr_method="flatten"
    )

    print("\nDownstream Utility Benchmark Results:")
    print(results_df.to_string(index=False))
    print("\nDownstream evaluation verified successfully.")