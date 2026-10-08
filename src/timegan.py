"""
PyTorch implementation of Time-Series Generative Adversarial Networks (TimeGAN).
Includes automatic accelerator detection (CUDA, Apple MPS, CPU), the four core sub-networks,
and the three-stage training pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict, Optional


def get_device() -> torch.device:
    """
    Selects the best available accelerator:
    1. CUDA (NVIDIA GPU / Colab)
    2. MPS (Apple Silicon Mac)
    3. CPU (Fallback)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------------------------------------------------------
# Sub-Networks
# -------------------------------------------------------------------------

class EmbeddingNetwork(nn.Module):
    """Maps feature space X (N, T, D) to latent space H (N, T, hidden_dim)."""
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_seq, _ = self.rnn(x)
        h = self.activation(self.fc(h_seq))
        return h


class RecoveryNetwork(nn.Module):
    """Maps latent space H (N, T, hidden_dim) back to feature space X_tilde (N, T, D)."""
    def __init__(self, hidden_dim: int, feature_dim: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, feature_dim)
        self.activation = nn.Sigmoid()  # Assumes features normalized to [0, 1]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x_seq, _ = self.rnn(h)
        x_tilde = self.activation(self.fc(x_seq))
        return x_tilde


class GeneratorNetwork(nn.Module):
    """Generates synthetic latent sequences E_hat from random noise Z."""
    def __init__(self, noise_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=noise_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        e_seq, _ = self.rnn(z)
        e_hat = self.activation(self.fc(e_seq))
        return e_hat


class SupervisorNetwork(nn.Module):
    """Enforces one-step-ahead temporal transition dynamics in latent space."""
    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        s_seq, _ = self.rnn(h)
        s = self.activation(self.fc(s_seq))
        return s


class DiscriminatorNetwork(nn.Module):
    """Classifies latent sequences as real (H) or synthetic (H_hat)."""
    def __init__(self, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        d_seq, _ = self.rnn(h)
        y_hat = self.fc(d_seq)  # Logits for BCEWithLogitsLoss
        return y_hat


# -------------------------------------------------------------------------
# Orchestrator & Multi-Stage Trainer
# -------------------------------------------------------------------------

class TimeGAN:
    def __init__(
        self,
        feature_dim: int,
        seq_len: int = 24,
        hidden_dim: int = 24,
        noise_dim: Optional[int] = None,
        num_layers: int = 3,
        lr: float = 1e-3,
        gamma: float = 1.0,
        device: Optional[torch.device] = None
    ):
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim or feature_dim
        self.gamma = gamma
        self.device = device or get_device()

        # Initialize Networks
        self.embedder = EmbeddingNetwork(feature_dim, hidden_dim, num_layers).to(self.device)
        self.recovery = RecoveryNetwork(hidden_dim, feature_dim, num_layers).to(self.device)
        self.generator = GeneratorNetwork(self.noise_dim, hidden_dim, num_layers).to(self.device)
        self.supervisor = SupervisorNetwork(hidden_dim, num_layers=max(1, num_layers - 1)).to(self.device)
        self.discriminator = DiscriminatorNetwork(hidden_dim, num_layers).to(self.device)

        # Loss Functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Optimizers
        self.opt_e = optim.Adam(list(self.embedder.parameters()) + list(self.recovery.parameters()), lr=lr)
        self.opt_s = optim.Adam(self.supervisor.parameters(), lr=lr)
        self.opt_g = optim.Adam(list(self.generator.parameters()) + list(self.supervisor.parameters()), lr=lr)
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=lr)

    def _sample_noise(self, batch_size: int) -> torch.Tensor:
        """Draws standard uniform noise vectors."""
        return torch.rand((batch_size, self.seq_len, self.noise_dim), device=self.device)

    def fit(
        self,
        data: np.ndarray,
        batch_size: int = 64,
        pretrain_epochs: int = 50,
        joint_epochs: int = 100
    ) -> Dict[str, list]:
        """Executes the full 3-phase TimeGAN training workflow."""
        tensor_x = torch.tensor(data, dtype=torch.float32)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        history = {"e_loss": [], "s_loss": [], "g_loss": [], "d_loss": []}

        # ----------------------------------------------------
        # Phase 1: Embedding Network Pretraining (Autoencoder)
        # ----------------------------------------------------
        print(f"\n[Phase 1/3] Embedding Pretraining ({pretrain_epochs} epochs) on {self.device}...")
        self.embedder.train()
        self.recovery.train()
        for epoch in range(1, pretrain_epochs + 1):
            epoch_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                self.opt_e.zero_grad()

                h = self.embedder(batch_x)
                x_tilde = self.recovery(h)

                loss_e = 10 * torch.sqrt(self.mse_loss(x_tilde, batch_x))
                loss_e.backward()
                self.opt_e.step()
                epoch_loss += loss_e.item()

            avg_loss = epoch_loss / len(loader)
            history["e_loss"].append(avg_loss)
            if epoch % max(1, pretrain_epochs // 5) == 0 or epoch == pretrain_epochs:
                print(f"  Epoch {epoch:03d}/{pretrain_epochs} - Loss_E: {avg_loss:.4f}")

        # ----------------------------------------------------
        # Phase 2: Supervised Pretraining
        # ----------------------------------------------------
        print(f"\n[Phase 2/3] Supervised Pretraining ({pretrain_epochs} epochs)...")
        self.supervisor.train()
        for epoch in range(1, pretrain_epochs + 1):
            epoch_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                self.opt_s.zero_grad()

                with torch.no_grad():
                    h = self.embedder(batch_x)

                h_hat_supervise = self.supervisor(h)
                # Next-step prediction loss: H[:, 1:, :] vs H_hat[:, :-1, :]
                loss_s = self.mse_loss(h_hat_supervise[:, :-1, :], h[:, 1:, :])

                loss_s.backward()
                self.opt_s.step()
                epoch_loss += loss_s.item()

            avg_loss = epoch_loss / len(loader)
            history["s_loss"].append(avg_loss)
            if epoch % max(1, pretrain_epochs // 5) == 0 or epoch == pretrain_epochs:
                print(f"  Epoch {epoch:03d}/{pretrain_epochs} - Loss_S: {avg_loss:.4f}")

        # ----------------------------------------------------
        # Phase 3: Joint Adversarial Training
        # ----------------------------------------------------
        print(f"\n[Phase 3/3] Joint Adversarial Training ({joint_epochs} epochs)...")
        self.generator.train()
        self.supervisor.train()
        self.discriminator.train()
        self.embedder.train()
        self.recovery.train()

        for epoch in range(1, joint_epochs + 1):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0

            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                current_batch_size = batch_x.size(0)

                # ====================
                # Train Generator
                # ====================
                self.opt_g.zero_grad()
                z = self._sample_noise(current_batch_size)

                e_hat = self.generator(z)
                h_hat = self.supervisor(e_hat)
                y_fake = self.discriminator(h_hat)

                # Unsupervised adversarial loss
                loss_g_u = self.bce_loss(y_fake, torch.ones_like(y_fake))

                # Supervised loss on generated dynamics
                loss_g_s = self.mse_loss(h_hat[:, :-1, :], e_hat[:, 1:, :])

                # Moments loss (captures mean and variance distribution matching)
                x_hat = self.recovery(h_hat)
                g_loss_v1 = torch.mean(torch.abs(torch.sqrt(torch.var(x_hat, dim=0) + 1e-6) - torch.sqrt(torch.var(batch_x, dim=0) + 1e-6)))
                g_loss_v2 = torch.mean(torch.abs(torch.mean(x_hat, dim=0) - torch.mean(batch_x, dim=0)))
                loss_g_v = g_loss_v1 + g_loss_v2

                # Combined Generator loss
                loss_g = loss_g_u + self.gamma * loss_g_s + 100 * torch.sqrt(loss_g_v)
                loss_g.backward()
                self.opt_g.step()

                # Train Embedder alongside Generator
                self.opt_e.zero_grad()
                h = self.embedder(batch_x)
                x_tilde = self.recovery(h)
                h_supervise = self.supervisor(h)

                loss_e_recon = 10 * torch.sqrt(self.mse_loss(x_tilde, batch_x))
                loss_e_s = self.mse_loss(h_supervise[:, :-1, :], h[:, 1:, :])
                loss_e = loss_e_recon + 0.1 * loss_e_s
                loss_e.backward()
                self.opt_e.step()

                # ====================
                # Train Discriminator
                # ====================
                self.opt_d.zero_grad()
                y_real = self.discriminator(h.detach())
                y_fake = self.discriminator(h_hat.detach())

                loss_d_real = self.bce_loss(y_real, torch.ones_like(y_real))
                loss_d_fake = self.bce_loss(y_fake, torch.zeros_like(y_fake))
                loss_d = loss_d_real + loss_d_fake

                if loss_d.item() > 0.15:  # Prevents discriminator from overpowering generator
                    loss_d.backward()
                    self.opt_d.step()

                epoch_g_loss += loss_g.item()
                epoch_d_loss += loss_d.item()

            avg_g = epoch_g_loss / len(loader)
            avg_d = epoch_d_loss / len(loader)
            history["g_loss"].append(avg_g)
            history["d_loss"].append(avg_d)

            if epoch % max(1, joint_epochs // 5) == 0 or epoch == joint_epochs:
                print(f"  Epoch {epoch:03d}/{joint_epochs} - Loss_G: {avg_g:.4f} | Loss_D: {avg_d:.4f}")

        return history

    def sample(self, num_samples: int) -> np.ndarray:
        """
        Generates synthetic transaction sequences.
        Output shape: (num_samples, seq_len, feature_dim) in [0, 1] scale.
        """
        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()

        with torch.no_grad():
            z = self._sample_noise(num_samples)
            e_hat = self.generator(z)
            h_hat = self.supervisor(e_hat)
            x_hat = self.recovery(h_hat)

        return x_hat.cpu().numpy()


# -------------------------------------------------------------------------
# Smoke Test / Verification
# -------------------------------------------------------------------------

if __name__ == "__main__":
    device = get_device()
    print("=" * 60)
    print(f"TimeGAN Verification running on: {device}")
    print("=" * 60)

    # Synthetic mock parameters
    N_SAMPLES = 128
    SEQ_LEN = 24
    N_FEATURES = 5

    # Generate synthetic input sequences
    mock_data = np.random.uniform(0.0, 1.0, size=(N_SAMPLES, SEQ_LEN, N_FEATURES)).astype(np.float32)

    # Initialize model
    model = TimeGAN(
        feature_dim=N_FEATURES,
        seq_len=SEQ_LEN,
        hidden_dim=16,
        num_layers=2,
        device=device
    )

    # Rapid sanity run: 2 epochs each
    print("\nExecuting dry run training loops...")
    hist = model.fit(mock_data, batch_size=32, pretrain_epochs=2, joint_epochs=2)

    # Test generation sampling
    synthetic_samples = model.sample(num_samples=10)
    print(f"\nSynthetic generation shape: {synthetic_samples.shape} (N, T, D)")
    print(f"Values range: [{synthetic_samples.min():.2f}, {synthetic_samples.max():.2f}]")
    assert synthetic_samples.shape == (10, SEQ_LEN, N_FEATURES)
    print("\nTimeGAN module verified successfully.")