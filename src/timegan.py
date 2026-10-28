"""
Time-Series Generative Adversarial Network (TimeGAN) Architecture.
Reference: Yoon, J., Jarrett, D., & van der Schaar, M. (NeurIPS 2019)
"Time-series Generative Adversarial Networks"

Optimized implementation featuring Tanh/LeakyReLU activations and standard
normal latent sampling to eliminate mode collapse and mean-line saturation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional


def get_device() -> torch.device:
    """Selects CUDA GPU if accessible; falls back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================================
# Core Sub-Network Modules
# ============================================================================

class EmbeddingNetwork(nn.Module):
    """
    Encodes raw time-series sequences into a low-dimensional latent space.
    e : X -> H
    """
    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_seq, _ = self.rnn(x)
        return self.activation(self.fc(h_seq))


class RecoveryNetwork(nn.Module):
    """
    Reconstructs original sequential feature coordinates from latent codes.
    r : H -> X_tilde
    """
    def __init__(self, hidden_dim: int, feature_dim: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, feature_dim)
        # Sigmoid is strictly applied at the final output to respect [0, 1] scaling
        self.activation = nn.Sigmoid()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x_seq, _ = self.rnn(h)
        return self.activation(self.fc(x_seq))


class GeneratorNetwork(nn.Module):
    """
    Generates synthetic latent sequences from stochastic noise vectors.
    g : Z -> E_hat
    """
    def __init__(self, noise_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=noise_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        e_seq, _ = self.rnn(z)
        return self.activation(self.fc(e_seq))


class SupervisorNetwork(nn.Module):
    """
    Enforces one-step-ahead temporal transition dynamics within the latent space.
    s : H_t -> H_{t+1}
    """
    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        s_seq, _ = self.rnn(h)
        return self.activation(self.fc(s_seq))


class DiscriminatorNetwork(nn.Module):
    """
    Evaluates authenticity across both real and synthetic latent representations.
    d : H -> Y
    """
    def __init__(self, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_intermediate = nn.Linear(hidden_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y_seq, _ = self.rnn(h)
        out = self.leaky_relu(self.fc_intermediate(y_seq))
        logits = self.fc_out(out)
        return logits


# ============================================================================
# Main TimeGAN Framework
# ============================================================================

class TimeGAN:
    """
    Complete TimeGAN framework coordinating 3-phase training and synthesis.
    """
    def __init__(
        self,
        feature_dim: int,
        seq_len: int = 24,
        hidden_dim: int = 32,
        num_layers: int = 3,
        lr: float = 1e-3,
        gamma: float = 1.0,
        device: Optional[torch.device] = None
    ):
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.noise_dim = feature_dim
        self.gamma = gamma
        self.device = device or get_device()

        # Instantiate 5 sub-networks
        self.embedder = EmbeddingNetwork(feature_dim, hidden_dim, num_layers).to(self.device)
        self.recovery = RecoveryNetwork(hidden_dim, feature_dim, num_layers).to(self.device)
        self.generator = GeneratorNetwork(self.noise_dim, hidden_dim, num_layers).to(self.device)
        self.supervisor = SupervisorNetwork(hidden_dim, num_layers=max(2, num_layers - 1)).to(self.device)
        self.discriminator = DiscriminatorNetwork(hidden_dim, num_layers).to(self.device)

        # Loss criteria
        self.mse_loss = nn.MSELoss()
        self.bce_logits_loss = nn.BCEWithLogitsLoss()

        # Optimizers
        self.e_opt = optim.Adam(list(self.embedder.parameters()) + list(self.recovery.parameters()), lr=lr)
        self.s_opt = optim.Adam(self.supervisor.parameters(), lr=lr)
        self.g_opt = optim.Adam(list(self.generator.parameters()) + list(self.supervisor.parameters()), lr=lr)
        self.d_opt = optim.Adam(self.discriminator.parameters(), lr=lr)

    def _sample_noise(self, batch_size: int) -> torch.Tensor:
        """Draws stochastic driving noise from standard normal distribution."""
        return torch.randn((batch_size, self.seq_len, self.noise_dim), device=self.device)

    def fit(
        self,
        data: np.ndarray,
        batch_size: int = 128,
        pretrain_epochs: int = 30,
        joint_epochs: int = 60
    ) -> Dict[str, List[float]]:
        """
        Executes the three-phase training pipeline:
        Phase 1: Embedding Autoencoder Pretraining
        Phase 2: Supervisor Transition Pretraining
        Phase 3: Joint Adversarial Game
        """
        tensor_data = torch.tensor(data, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        history: Dict[str, List[float]] = {
            "e_loss": [],
            "s_loss": [],
            "g_loss": [],
            "d_loss": []
        }

        # --------------------------------------------------------------------
        # Phase 1: Embedding & Recovery Autoencoder Pretraining
        # --------------------------------------------------------------------
        print(f"\n[Phase 1/3] Autoencoder Pretraining ({pretrain_epochs} Epochs)...")
        for epoch in range(1, pretrain_epochs + 1):
            total_e_loss = 0.0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)

                self.e_opt.zero_grad()
                h = self.embedder(batch_x)
                x_tilde = self.recovery(h)

                loss_e = 10.0 * torch.sqrt(self.mse_loss(x_tilde, batch_x) + 1e-8)
                loss_e.backward()
                nn.utils.clip_grad_norm_(list(self.embedder.parameters()) + list(self.recovery.parameters()), 1.0)
                self.e_opt.step()

                total_e_loss += loss_e.item()

            avg_e_loss = total_e_loss / len(dataloader)
            history["e_loss"].append(avg_e_loss)
            if epoch % 5 == 0 or epoch == pretrain_epochs:
                print(f"  Epoch {epoch:03d}/{pretrain_epochs:03d} - Loss_E: {avg_e_loss:.5f}")

        # --------------------------------------------------------------------
        # Phase 2: Supervisor Latent Transition Pretraining
        # --------------------------------------------------------------------
        print(f"\n[Phase 2/3] Supervisor Latent Pretraining ({pretrain_epochs} Epochs)...")
        for epoch in range(1, pretrain_epochs + 1):
            total_s_loss = 0.0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)

                self.s_opt.zero_grad()
                with torch.no_grad():
                    h = self.embedder(batch_x)

                h_hat_supervise = self.supervisor(h)

                # Next-step latent prediction: align h[:, 1:] with h_hat[:, :-1]
                loss_s = self.mse_loss(h_hat_supervise[:, :-1, :], h[:, 1:, :])
                loss_s.backward()
                nn.utils.clip_grad_norm_(self.supervisor.parameters(), 1.0)
                self.s_opt.step()

                total_s_loss += loss_s.item()

            avg_s_loss = total_s_loss / len(dataloader)
            history["s_loss"].append(avg_s_loss)
            if epoch % 5 == 0 or epoch == pretrain_epochs:
                print(f"  Epoch {epoch:03d}/{pretrain_epochs:03d} - Loss_S: {avg_s_loss:.5f}")

        # --------------------------------------------------------------------
        # Phase 3: Joint Adversarial Training
        # --------------------------------------------------------------------
        print(f"\n[Phase 3/3] Joint Adversarial Training ({joint_epochs} Epochs)...")
        for epoch in range(1, joint_epochs + 1):
            total_g_loss = 0.0
            total_d_loss = 0.0

            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                curr_bs = batch_x.size(0)

                # ----------------- Train Generator & Supervisor -----------------
                for _ in range(2):
                    self.g_opt.zero_grad()
                    z = self._sample_noise(curr_bs)

                    e_hat = self.generator(z)
                    h_hat = self.supervisor(e_hat)
                    x_hat = self.recovery(h_hat)

                    y_fake = self.discriminator(h_hat)

                    # Adversarial unsupervised loss
                    loss_u = self.bce_logits_loss(y_fake, torch.ones_like(y_fake))

                    # Supervised step loss on synthesized codes
                    loss_s_fake = self.mse_loss(h_hat[:, :-1, :], e_hat[:, 1:, :])

                    # Feature moment matching (mean and variance regularization)
                    loss_moments = (
                        torch.mean(torch.abs(torch.mean(x_hat, dim=0) - torch.mean(batch_x, dim=0))) +
                        torch.mean(torch.abs(torch.std(x_hat, dim=0) - torch.std(batch_x, dim=0)))
                    )

                    loss_g = loss_u + self.gamma * loss_s_fake + 10.0 * loss_moments
                    loss_g.backward()
                    nn.utils.clip_grad_norm_(list(self.generator.parameters()) + list(self.supervisor.parameters()), 1.0)
                    self.g_opt.step()

                # Train Embedder & Recovery joint reconstruction
                self.e_opt.zero_grad()
                h = self.embedder(batch_x)
                x_tilde = self.recovery(h)
                h_hat_supervise = self.supervisor(h)

                loss_e0 = 10.0 * torch.sqrt(self.mse_loss(x_tilde, batch_x) + 1e-8)
                loss_s_real = self.mse_loss(h_hat_supervise[:, :-1, :], h[:, 1:, :])
                loss_er = loss_e0 + 0.1 * loss_s_real

                loss_er.backward()
                nn.utils.clip_grad_norm_(list(self.embedder.parameters()) + list(self.recovery.parameters()), 1.0)
                self.e_opt.step()

                # ----------------- Train Discriminator -----------------
                self.d_opt.zero_grad()
                with torch.no_grad():
                    h_real = self.embedder(batch_x)
                    z_disc = self._sample_noise(curr_bs)
                    e_hat_disc = self.generator(z_disc)
                    h_fake = self.supervisor(e_hat_disc)

                y_real = self.discriminator(h_real)
                y_fake = self.discriminator(h_fake)

                loss_d_real = self.bce_logits_loss(y_real, torch.ones_like(y_real))
                loss_d_fake = self.bce_logits_loss(y_fake, torch.zeros_like(y_fake))
                loss_d = loss_d_real + loss_d_fake

                loss_d.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.d_opt.step()

                total_g_loss += loss_g.item()
                total_d_loss += loss_d.item()

            avg_g_loss = total_g_loss / (len(dataloader) * 2)
            avg_d_loss = total_d_loss / len(dataloader)
            history["g_loss"].append(avg_g_loss)
            history["d_loss"].append(avg_d_loss)

            if epoch % 5 == 0 or epoch == joint_epochs:
                print(f"  Epoch {epoch:03d}/{joint_epochs:03d} - Loss_G: {avg_g_loss:.5f} | Loss_D: {avg_d_loss:.5f}")

        return history

    def sample(self, num_samples: int) -> np.ndarray:
        """
        Generates synthetic sequential data points: Z -> g(Z) -> s(H) -> r(H) -> X_hat
        """
        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()

        with torch.no_grad():
            z = self._sample_noise(num_samples)
            e_hat = self.generator(z)
            h_hat = self.supervisor(e_hat)
            x_hat = self.recovery(h_hat)

        synthetic_data = x_hat.cpu().numpy()

        self.generator.train()
        self.supervisor.train()
        self.recovery.train()

        return synthetic_data