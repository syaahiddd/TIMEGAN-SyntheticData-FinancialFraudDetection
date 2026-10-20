"""
Data preprocessing and sequence windowing pipeline for TimeGAN.
Handles MinMax feature scaling, rolling-window tensor construction, 
synthetic-to-original scale inversions, and credit card fraud dataset ingestion.
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
        """
        Args:
            seq_len: Number of temporal steps in each sliding window (T).
            scaler_range: Min-Max normalization bounds (default [0, 1]).
        """
        self.seq_len = seq_len
        self.scaler = MinMaxScaler(feature_range=scaler_range)
        self.feature_names: Optional[List[str]] = None

    def fit_transform_tabular(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """
        Extracts selected numerical features and normalizes them across the dataset.
        """
        self.feature_names = feature_cols
        data_matrix = df[feature_cols].values.astype(np.float32)
        scaled_data = self.scaler.fit_transform(data_matrix)
        return scaled_data

    def transform_tabular(self, df: pd.DataFrame) -> np.ndarray:
        """
        Applies pre-fitted scaler to unseen test tabular data.
        """
        if self.feature_names is None:
            raise ValueError("Scaler has not been fitted yet. Call fit_transform_tabular first.")
        data_matrix = df[self.feature_names].values.astype(np.float32)
        return self.scaler.transform(data_matrix)

    def inverse_transform_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """
        Reconstructs original feature scales from synthetic 3D arrays of shape (N, T, D).
        """
        if self.feature_names is None:
            raise ValueError("Scaler has not been fitted yet.")
        n_samples, t_steps, n_feats = sequences.shape
        reshaped = sequences.reshape(-1, n_feats)
        denorm = self.scaler.inverse_transform(reshaped)
        return denorm.reshape(n_samples, t_steps, n_feats)

    def build_sliding_windows(self, scaled_array: np.ndarray, stride: int = 1) -> np.ndarray:
        """
        Extracts rolling sequences from a 2D matrix of shape (total_steps, n_features).
        Returns a 3D NumPy array of shape (N, seq_len, n_features).
        """
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


def generate_mock_financial_data(n_records: int = 1000) -> pd.DataFrame:
    """
    Generates synthetic transaction logs to verify the pipeline functionality
    before you load large external CSV datasets.
    """
    np.random.seed(42)
    time_steps = np.arange(n_records)

    # Cyclical spending and running balances
    amounts = np.abs(np.sin(time_steps * 0.05) * 100 + np.random.normal(50, 10, n_records))
    balance_orig = np.maximum(0, 10000 - np.cumsum(amounts * 0.05) + np.random.normal(0, 20, n_records))
    balance_dest = np.maximum(0, np.cumsum(amounts * 0.04) + np.random.normal(0, 20, n_records))
    
    # Anonymized latent features (like V1, V2 in credit card datasets)
    v1 = np.cos(time_steps * 0.02) + np.random.normal(0, 0.1, n_records)
    v2 = np.sin(time_steps * 0.03) + np.random.normal(0, 0.1, n_records)

    df = pd.DataFrame({
        "step": time_steps,
        "amount": amounts,
        "oldbalanceOrg": balance_orig,
        "newbalanceDest": balance_dest,
        "V1": v1,
        "V2": v2
    })
    return df


def load_and_preprocess_creditcard(
    csv_path: str,
    seq_len: int = 24,
    stride: int = 1,
    subsample_limit: Optional[int] = 50000
) -> Tuple[np.ndarray, np.ndarray, FinancialTimeSeriesLoader]:
    """
    Ingests ULB Credit Card dataset, sorts temporally, engineers delta_time,
    selects top predictive fraud features, scales to [0, 1], and slices into 3D tensors.
    """
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Sort temporally
    df = df.sort_values(by="Time").reset_index(drop=True)

    # 2. Subsample to prevent memory exhaustion if specified
    if subsample_limit and len(df) > subsample_limit:
        print(f"Subsampling to first {subsample_limit} continuous temporal events...")
        df = df.iloc[:subsample_limit].reset_index(drop=True)

    # 3. Temporal delta
    df["delta_time"] = df["Time"].diff().fillna(0.0)

    # 4. Top predictive features
    feature_cols = ["V14", "V12", "V17", "V10", "V11", "V4", "Amount", "delta_time"]

    # 5. Extract targets for downstream benchmarks
    targets = df["Class"].values.astype(int)

    # 6. Fit & transform features
    loader = FinancialTimeSeriesLoader(seq_len=seq_len, scaler_range=(0.0, 1.0))
    scaled_matrix = loader.fit_transform_tabular(df, feature_cols=feature_cols)

    # 7. Generate 3D sequences
    sequences = loader.build_sliding_windows(scaled_matrix, stride=stride)

    # Window labels: label sequence as fraud if any transaction in window is fraudulent
    window_labels = []
    for i in range(0, len(targets) - seq_len + 1, stride):
        window_labels.append(int(np.any(targets[i : i + seq_len] == 1)))
    window_labels = np.array(window_labels, dtype=int)

    print(f"Constructed 3D Sequences shape: {sequences.shape} (N, T, D)")
    print(f"Window Fraud Distribution: Normal={np.sum(window_labels == 0)}, Fraud={np.sum(window_labels == 1)}")

    return sequences, window_labels, loader


if __name__ == "__main__":
    print("Running pipeline verification test in src/data_loader.py...")

    # 1. Generate test tabular dataframe
    raw_df = generate_mock_financial_data(n_records=1000)
    features = ["amount", "oldbalanceOrg", "newbalanceDest", "V1", "V2"]
    
    # 2. Instantiate loader
    seq_length = 24
    loader = FinancialTimeSeriesLoader(seq_len=seq_length, scaler_range=(0.0, 1.0))
    
    # 3. Scale features
    scaled_data = loader.fit_transform_tabular(raw_df, feature_cols=features)
    
    # 4. Generate sliding windows
    tensor_3d = loader.build_sliding_windows(scaled_data, stride=1)
    
    # 5. Verify inverse scaling preserves shape
    reconstructed = loader.inverse_transform_sequences(tensor_3d)

    print(f"-> Tabular input shape:     {raw_df[features].shape}")
    print(f"-> Scaled 2D matrix shape:  {scaled_data.shape}")
    print(f"-> 3D Window tensor shape:  {tensor_3d.shape}  [Samples, Seq_Len, Features]")
    print(f"-> Tensor value range:      [{tensor_3d.min():.2f}, {tensor_3d.max():.2f}]")
    print(f"-> Inversion shape match:   {reconstructed.shape == tensor_3d.shape}")
    print("Verification passed.")