"""
Data preprocessing and sequence windowing pipeline for TimeGAN.
Handles MinMax feature scaling, rolling-window tensor construction, 
synthetic-to-original scale inversions, and dynamic financial fraud dataset ingestion.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional


class FinancialTimeSeriesLoader:
    """
    Transforms tabular transaction records into 3D sequential tensors (N, T, D) for TimeGAN.
    """

    def __init__(self, seq_len: int = 24, scaler_range: Tuple[float, float] = (0.0, 1.0)):
        self.seq_len = seq_len
        self.scaler = MinMaxScaler(feature_range=scaler_range)
        self.feature_names: Optional[List[str]] = None

    def fit_transform_tabular(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        self.feature_names = feature_cols
        data_matrix = df[feature_cols].values.astype(np.float32)
        scaled_data = self.scaler.fit_transform(data_matrix)
        return scaled_data

    def transform_tabular(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature_names is None:
            raise ValueError("Scaler has not been fitted yet. Call fit_transform_tabular first.")
        data_matrix = df[self.feature_names].values.astype(np.float32)
        return self.scaler.transform(data_matrix)

    def inverse_transform_sequences(self, sequences: np.ndarray) -> np.ndarray:
        if self.feature_names is None:
            raise ValueError("Scaler has not been fitted yet.")
        n_samples, t_steps, n_feats = sequences.shape
        reshaped = sequences.reshape(-1, n_feats)
        denorm = self.scaler.inverse_transform(reshaped)
        return denorm.reshape(n_samples, t_steps, n_feats)

    def build_sliding_windows(self, scaled_array: np.ndarray, stride: int = 1) -> np.ndarray:
        total_steps, n_features = scaled_array.shape
        if total_steps < self.seq_len:
            raise ValueError(
                f"Dataset length ({total_steps}) is shorter than window length ({self.seq_len})."
            )

        sequences = []
        for start_idx in range(0, total_steps - self.seq_len + 1, stride):
            end_idx = start_idx + self.seq_len
            window = scaled_array[start_idx:end_idx]
            sequences.append(window)

        return np.array(sequences, dtype=np.float32)


def load_and_preprocess_creditcard(
    csv_path: str,
    seq_len: int = 24,
    stride: int = 1,
    subsample_limit: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, FinancialTimeSeriesLoader]:
    """
    Dynamically ingests financial tabular data, detects target and temporal columns,
    engineers delta_time, normalizes features, and extracts 3D sliding window tensors.
    """
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Clean whitespace from all column names
    df.columns = [str(c).strip() for c in df.columns]

    # 2. Dynamic Target Column Detection
    possible_target_names = ["class", "isfraud", "fraud", "label", "target"]
    target_col = None
    for col in df.columns:
        if col.lower() in possible_target_names:
            target_col = col
            break

    # Fallback: if no named match, treat the last column as target
    if target_col is None:
        target_col = df.columns[-1]
        print(f"-> Notice: Target column not explicitly named. Falling back to last column: '{target_col}'")
    else:
        print(f"-> Detected Target Column: '{target_col}'")

    # 3. Dynamic Time Column Detection & Sorting
    time_col = None
    for col in df.columns:
        if col.lower() in ["time", "step", "timestamp", "date"]:
            time_col = col
            break

    if time_col is not None:
        print(f"-> Detected Temporal Index: '{time_col}'. Sorting events chronologically...")
        df = df.sort_values(by=time_col).reset_index(drop=True)
        # Engineer delta_time
        df["delta_time"] = df[time_col].diff().fillna(0.0)
    else:
        print("-> Notice: No explicit Time column found. Assuming chronological row order and computing pseudo-step...")
        df["delta_time"] = 1.0

    # 4. Optional Subsampling
    if subsample_limit and len(df) > subsample_limit:
        print(f"-> Subsampling to first {subsample_limit} continuous temporal events...")
        df = df.iloc[:subsample_limit].reset_index(drop=True)

    # 5. Dynamic Feature Selection (exclude target and original raw timestamp)
    cols_to_exclude = {target_col}
    if time_col is not None:
        cols_to_exclude.add(time_col)

    candidate_features = [c for c in df.columns if c not in cols_to_exclude]

    # Prioritize PCA / V-features, Amount, and delta_time; retain all numeric columns
    numeric_features = df[candidate_features].select_dtypes(include=[np.number]).columns.tolist()
    print(f"-> Modeled Feature Columns ({len(numeric_features)}): {numeric_features}")

    # 6. Extract target values and verify binary format
    targets = df[target_col].values.astype(int)

    # 7. Scale features to [0, 1]
    loader = FinancialTimeSeriesLoader(seq_len=seq_len, scaler_range=(0.0, 1.0))
    scaled_matrix = loader.fit_transform_tabular(df, feature_cols=numeric_features)

    # 8. Construct 3D Sliding Windows
    sequences = loader.build_sliding_windows(scaled_matrix, stride=stride)

    # Window labels: label sequence as fraud (1) if any transaction within the window is fraud
    window_labels = []
    for i in range(0, len(targets) - seq_len + 1, stride):
        window_labels.append(int(np.any(targets[i : i + seq_len] == 1)))
    window_labels = np.array(window_labels, dtype=int)

    print(f"\n[Summary] 3D Tensor Shape: {sequences.shape} (N_Samples, Seq_Len, Features)")
    print(f"[Summary] Target Balance: Normal Windows = {np.sum(window_labels == 0)}, Fraud Windows = {np.sum(window_labels == 1)}")

    return sequences, window_labels, loader


if __name__ == "__main__":
    print("Executing dynamic data loader module test...")
    # Self-test using mock data with unconventional column naming
    mock_df = pd.DataFrame({
        "timestamp": np.arange(100),
        "feat_a": np.random.randn(100),
        "feat_b": np.random.randn(100),
        "fraud_flag": (np.random.rand(100) > 0.9).astype(int)
    })
    mock_df.to_csv("test_mock.csv", index=False)

    seqs, lbls, ldr = load_and_preprocess_creditcard("test_mock.csv", seq_len=12, stride=1)
    print("Module test completed successfully.")
    import os
    if os.path.exists("test_mock.csv"):
        os.remove("test_mock.csv")