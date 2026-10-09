"""
Tri-Pillar evaluation metrics for synthetic financial time-series:
1. Fidelity: PCA & t-SNE visual distribution alignment
2. Temporal Realism: Discriminative score via a 2-layer post-hoc GRU
3. Privacy Preservation: Distance to Closest Record (DCR) empirical leakage
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Optional


# -------------------------------------------------------------------------
# 1. Fidelity: Dimensionality Reduction (PCA & t-SNE)
# -------------------------------------------------------------------------

def evaluate_distribution_overlap(
    real_data: np.ndarray,
    synth_data: np.ndarray,
    n_samples: int = 1000,
    save_path: Optional[str] = None
) -> None:
    """
    Flattens 3D sequences into 2D and generates PCA and t-SNE projections 
    to visually assess distribution overlap.
    """
    n_eval = min(len(real_data), len(synth_data), n_samples)
    idx_real = np.random.choice(len(real_data), n_eval, replace=False)
    idx_synth = np.random.choice(len(synth_data), n_eval, replace=False)

    real_flat = real_data[idx_real].reshape(n_eval, -1)
    synth_flat = synth_data[idx_synth].reshape(n_eval, -1)

    combined = np.vstack([real_flat, synth_flat])
    labels = np.array([0] * n_eval + [1] * n_eval)

    # 1. PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(combined)

    # 2. t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(combined)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot PCA
    axes[0].scatter(pca_results[labels == 0, 0], pca_results[labels == 0, 1], c='royalblue', alpha=0.35, label='Real')
    axes[0].scatter(pca_results[labels == 1, 0], pca_results[labels == 1, 1], c='crimson', alpha=0.35, label='Synthetic')
    axes[0].set_title(f"PCA Projection (Explained Var: {pca.explained_variance_ratio_.sum():.2%})")
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Plot t-SNE
    axes[1].scatter(tsne_results[labels == 0, 0], tsne_results[labels == 0, 1], c='royalblue', alpha=0.35, label='Real')
    axes[1].scatter(tsne_results[labels == 1, 0], tsne_results[labels == 1, 1], c='crimson', alpha=0.35, label='Synthetic')
    axes[1].set_title("t-SNE Manifold Representation")
    axes[1].set_xlabel("Dimension 1")
    axes[1].set_ylabel("Dimension 2")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Distribution plot saved to: {save_path}")
    plt.close()


# -------------------------------------------------------------------------
# 2. Fidelity: Post-Hoc Discriminative Score
# -------------------------------------------------------------------------

class PostHocDiscriminator(nn.Module):
    """Auxiliary classifier to measure synthetic distinguishability."""
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        logits = self.fc(out[:, -1, :])  # Final temporal hidden state
        return logits


def compute_discriminative_score(
    real_data: np.ndarray,
    synth_data: np.ndarray,
    epochs: int = 15,
    batch_size: int = 64,
    device: Optional[torch.device] = None
) -> float:
    """
    Trains a separate GRU classifier to predict whether a sequence is real (0) or synthetic (1).
    Returns absolute difference from random guess (|accuracy - 0.5|).
    Ideal score: 0.00 (completely indistinguishable).
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    n_samples = min(len(real_data), len(synth_data))

    x = np.concatenate([real_data[:n_samples], synth_data[:n_samples]], axis=0)
    y = np.array([0] * n_samples + [1] * n_samples, dtype=np.float32)

    # 80/20 train/test split
    indices = np.random.permutation(len(x))
    split = int(len(x) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(torch.tensor(x[train_idx], dtype=torch.float32), torch.tensor(y[train_idx]))
    test_ds = TensorDataset(torch.tensor(x[test_idx], dtype=torch.float32), torch.tensor(y[test_idx]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = PostHocDiscriminator(input_dim=real_data.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device).unsqueeze(1)
            preds = (torch.sigmoid(model(bx)) >= 0.5).float()
            correct += (preds == by).sum().item()
            total += by.size(0)

    accuracy = correct / total
    disc_score = abs(accuracy - 0.5)
    return disc_score


# -------------------------------------------------------------------------
# 3. Privacy: Distance to Closest Record (DCR)
# -------------------------------------------------------------------------

def compute_dcr_metrics(
    train_data: np.ndarray,
    synth_data: np.ndarray,
    test_data: np.ndarray,
    n_sample: int = 500
) -> Dict[str, float]:
    """
    Computes minimum Euclidean distance from synthetic samples to real training vs. holdout test records.
    If DCR_train << DCR_test, the generator is memorizing the training data (privacy leak).
    """
    n_eval = min(len(train_data), len(synth_data), len(test_data), n_sample)

    train_flat = train_data[:n_eval].reshape(n_eval, -1)
    synth_flat = synth_data[:n_eval].reshape(n_eval, -1)
    test_flat = test_data[:n_eval].reshape(n_eval, -1)

    # Pairwise minimum distance from synthetic to train
    dists_train = []
    dists_test = []

    for s in synth_flat:
        d_tr = np.min(np.linalg.norm(train_flat - s, axis=1))
        d_te = np.min(np.linalg.norm(test_flat - s, axis=1))
        dists_train.append(d_tr)
        dists_test.append(d_te)

    dcr_train_mean = float(np.mean(dists_train))
    dcr_test_mean = float(np.mean(dists_test))
    
    # 5th percentile reveals extreme memorization risks
    dcr_5th_pct_train = float(np.percentile(dists_train, 5))

    return {
        "dcr_train_mean": dcr_train_mean,
        "dcr_test_mean": dcr_test_mean,
        "dcr_ratio": dcr_train_mean / (dcr_test_mean + 1e-8),
        "dcr_5th_percentile_train": dcr_5th_pct_train
    }


# -------------------------------------------------------------------------
# Verification Runner
# -------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing metrics module verification...")

    # Mock real, synthetic, and test sequences (N=100, T=24, D=5)
    np.random.seed(42)
    mock_real = np.random.uniform(0.1, 0.9, size=(100, 24, 5)).astype(np.float32)
    mock_synth = mock_real + np.random.normal(0, 0.05, size=(100, 24, 5)).astype(np.float32)
    mock_test = np.random.uniform(0.1, 0.9, size=(100, 24, 5)).astype(np.float32)

    # 1. Test DCR
    dcr_results = compute_dcr_metrics(mock_real, mock_synth, mock_test, n_sample=50)
    print("\n[Privacy] DCR Results:")
    for k, v in dcr_results.items():
        print(f"  {k}: {v:.4f}")

    # 2. Test Discriminative Score
    disc_score = compute_discriminative_score(mock_real, mock_synth, epochs=2, batch_size=32)
    print(f"\n[Fidelity] Post-Hoc Discriminative Score (|Acc - 0.5|): {disc_score:.4f}")

    # 3. Test PCA / t-SNE Plotting
    evaluate_distribution_overlap(mock_real, mock_synth, n_samples=50, save_path="results/figures/test_overlap.png")

    print("\nAll metrics functions verified successfully.")