"""Model architecture, losses, metrics, and dataset wrappers for VecMagNet.

Contains the published AttentionDipoleModel ("DipoleNet") and everything it
needs; the legacy/experimental architectures from the research repository are
not part of this release.
"""
import math
import os
import re
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class h5py_dset(torch.utils.data.Dataset):
    """Dataset wrapper that yields HDF5 entries as float tensors."""

    def __init__(self, h5_file, dataset_name):
        self.h5_file = h5py.File(h5_file, "r")
        self.keys = list(self.h5_file.keys())
        self.dataset = self.h5_file[dataset_name]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx], dtype=torch.float32)

    def get_keys(self):
        return self.keys


class Dataloader(torch.utils.data.Dataset):
    """Simple dataset and dataloader factory for paired tensors."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self._dataset = torch.utils.data.TensorDataset(self.x, self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def shuffle(self):
        perm = torch.randperm(len(self.x))
        self.x = self.x[perm]
        self.y = self.y[perm]

    def dataset(self):
        return torch.utils.data.TensorDataset(self.x, self.y)

    def dataloaders(self, batch_size=32, train_val_split=0.8):
        dataset = self._dataset
        train_size = int(train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        return (
            torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        )


# Torch error metrics

def declination_error(
    output: torch.Tensor,
    target: torch.Tensor,
    *,
    horizontal_threshold: float = 1e-3,
    return_fraction: bool = False,
):
    """Compute the absolute declination error while masking near-zero targets."""

    horizontal_magnitude_target = torch.norm(target[:, :2], dim=1)  # horizontal magnetic moment magnitude (AÂ·m^2)
    valid_mask = horizontal_magnitude_target > horizontal_threshold

    if not torch.any(valid_mask):
        warnings.warn(
            "Declination error skipped: all samples have negligible horizontal magnetic moment.",
            RuntimeWarning,
            stacklevel=2,
        )
        mean_declination_error = output.new_tensor(0.0)
        valid_fraction = output.new_tensor(0.0)
        if return_fraction:
            return mean_declination_error, valid_fraction
        return mean_declination_error

    masked_output = output[valid_mask]
    masked_target = target[valid_mask]

    output_declination = torch.atan2(masked_output[:, 1], masked_output[:, 0])
    target_declination = torch.atan2(masked_target[:, 1], masked_target[:, 0])

    declination_diff = (output_declination - target_declination + torch.pi) % (2 * torch.pi) - torch.pi
    mean_declination_error = torch.mean(torch.abs(declination_diff))
    valid_fraction = valid_mask.float().mean()

    if return_fraction:
        return mean_declination_error, valid_fraction
    return mean_declination_error


def inclination_error(output, target):
    """Return the mean absolute inclination difference."""

    output_inclination = torch.atan2(output[:, 2], torch.norm(output[:, :2], dim=1))
    target_inclination = torch.atan2(target[:, 2], torch.norm(target[:, :2], dim=1))

    inclination_diff = (output_inclination - target_inclination + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.mean(torch.abs(inclination_diff))


def angle_loss(output, target):
    """Sum of inclination and declination losses."""

    inclination_loss = inclination_error(output, target)
    declination_loss = declination_error(output, target)
    return inclination_loss + declination_loss


def mag_error(output, target):
    """Mean absolute magnetic magnitude error."""

    output_mag = torch.norm(output, dim=1)
    target_mag = torch.norm(target, dim=1)
    return torch.mean(torch.abs(output_mag - target_mag))


def mag_p_error(output, target):
    """Mean absolute percentage magnetic magnitude error."""

    output_mag = torch.norm(output, dim=1)
    target_mag = torch.norm(target, dim=1)
    return torch.mean(torch.abs((output_mag - target_mag) / (target_mag + 1e-8)))


def mag_diff(output_mag, target_mag):
    """Mean absolute percentage difference between predicted and target magnitudes."""

    return torch.mean(torch.abs((output_mag - target_mag) / (target_mag + 1e-8)))


def combined_loss(pred, target, alpha=1.1, beta=1.0):
    """Blend mean squared error with angular disagreement."""

    mse = F.mse_loss(pred, target)

    pred_norm = F.normalize(pred, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)
    cos_sim = (pred_norm * target_norm).sum(dim=1)
    angle_penalty = torch.acos(torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)).mean()

    return alpha * mse + beta * angle_penalty


def full_vector_loss(pred, target, delta=1.0, alpha=2.5, beta=5.0):
    """Blend Huber loss with angular disagreement."""

    huber = F.huber_loss(pred, target, delta=delta)

    pred_norm = F.normalize(pred, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)
    cos_sim = (pred_norm * target_norm).sum(dim=1)
    angle_penalty = (1.0 - cos_sim).mean()  # bounded [0, 2]; no acos/normalization gradient blow-up

    return alpha * huber + beta * angle_penalty




class AttentionPooling(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        weights = F.softmax(self.attention(x_flat), dim=1)  # [B, HW, 1]
        return (x_flat * weights).sum(dim=1)  # [B, C]
    
class DipoleNet(nn.Module):
    def __init__(self, backbone_dim=512):
        super().__init__()
        self.direction_head = nn.Linear(backbone_dim, 3)
        self.magnitude_head = nn.Sequential(
            nn.Linear(backbone_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)  # predicts log-magnitude
        )
    def forward(self, features):
        direction = F.normalize(self.direction_head(features), dim=-1)
        log_mag = self.magnitude_head(features)
        magnitude = torch.exp(log_mag)
        moment = direction * magnitude
        return moment, direction, log_mag
    
def dipole_field_torch(m: torch.Tensor, r_grid: torch.Tensor) -> torch.Tensor:
    """Vectorized dipole field B(r) = (μ₀/4π)[3(m·r̂)r̂ - m]/|r|³.

    Args:
        m: Dipole moments [B, 3] in A·m².
        r_grid: Spatial coordinates [H, W, 3] in metres.
    Returns:
        Field tensor [B, 3, H, W] in Tesla.
    """
    MU0_OVER_4PI = 1e-7  # T·m/A
    H, W, _ = r_grid.shape
    N = H * W
    r = r_grid.view(N, 3)
    r_norm = r.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    r_hat = r / r_norm          # [N, 3]
    r_norm3 = r_norm ** 3       # [N, 1]

    m_dot_rhat = torch.einsum("bi,ni->bn", m, r_hat)          # [B, N]
    term1 = 3.0 * m_dot_rhat.unsqueeze(-1) * r_hat.unsqueeze(0)  # [B, N, 3]
    term2 = m.unsqueeze(1).expand(-1, N, -1)                   # [B, N, 3]
    field = MU0_OVER_4PI * (term1 - term2) / r_norm3.unsqueeze(0)  # [B, N, 3]
    return field.permute(0, 2, 1).reshape(m.shape[0], 3, H, W)


def forward_consistency_loss(pred_moment, input_field, r_grid):
    B_pred = dipole_field_torch(pred_moment, r_grid)
    return F.mse_loss(B_pred, input_field)


def physics_refine(
    moment_init: torch.Tensor,
    B_obs: torch.Tensor,
    r_grid: torch.Tensor,
    *,
    n_steps: int = 50,
    lr: float = 1e-3,
) -> torch.Tensor:
    """Refine a single-dipole moment estimate by minimising the forward-model residual.

    Starting from the model's initial prediction, runs Adam gradient descent on
    ||B_dipole(m; r_grid) - B_obs||² to correct errors in both magnitude and
    direction.  Works best for N=1 dipole samples where the single-dipole forward
    model is exact; for N>1 it finds the best-fitting equivalent dipole.

    Args:
        moment_init: Initial moment vector [3] from the neural network.
        B_obs: Observed magnetic field [3, H, W] in physical units (e.g. Tesla).
        r_grid: Measurement grid coordinates [H, W, 3] in metres.
        n_steps: Number of Adam optimisation steps.
        lr: Adam learning rate.

    Returns:
        Refined moment vector [3], detached from the computation graph.
    """
    device = moment_init.device
    m = moment_init.clone().float().detach().requires_grad_(True)
    optimiser = torch.optim.Adam([m], lr=lr)
    B_obs_batch = B_obs.unsqueeze(0).float().to(device)   # [1, 3, H, W]
    r_grid_dev = r_grid.to(device)
    for _ in range(n_steps):
        optimiser.zero_grad()
        B_pred = dipole_field_torch(m.unsqueeze(0), r_grid_dev)
        loss = F.mse_loss(B_pred, B_obs_batch)
        loss.backward()
        optimiser.step()
    # Relative residual: RMSE / RMS(B_obs) — scale-invariant goodness-of-fit.
    # Near zero → field is consistent with a single dipole.
    # Large (e.g. >0.15) → multi-dipole superposition; refinement should be discarded.
    with torch.no_grad():
        B_final = dipole_field_torch(m.unsqueeze(0), r_grid_dev)
        rmse = F.mse_loss(B_final, B_obs_batch).sqrt()
        rms_obs = B_obs_batch.pow(2).mean().sqrt().clamp(min=1e-30)
        rel_residual = (rmse / rms_obs).item()
    return m.detach(), rel_residual


def make_measurement_grid(
    grid_size: int = 100,
    extent: float = 1.0,
    z_offset: float = 0.1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Construct a uniform measurement grid for the physics-consistency loss.

    Coordinates are in the same normalized [0, 1] units used during data
    generation (not metres).  The grid is centred at the dipole origin, so x
    and y run from -extent/2 to +extent/2.  For a dipole centred at (0.5,
    0.5) in a [0, 1] image, extent=1.0 covers the full image footprint.
    z_offset should match the mapheight used during dataset generation.

    Args:
        grid_size: Number of grid points per side.
        extent: Side length of the grid in generation units (default 1.0 =
            full [0, 1] image span).
        z_offset: Height of the measurement plane above the dipole in
            generation units (should equal dataset mapheight, typically 0.1).
        device: Target device.
    Returns:
        Spatial coordinate tensor [grid_size, grid_size, 3].
    """
    xs = torch.linspace(-extent / 2, extent / 2, grid_size)
    ys = torch.linspace(-extent / 2, extent / 2, grid_size)
    xg, yg = torch.meshgrid(xs, ys, indexing="ij")
    zg = torch.full_like(xg, z_offset)
    r_grid = torch.stack([xg, yg, zg], dim=-1)  # [H, W, 3]
    if device is not None:
        r_grid = r_grid.to(device)
    return r_grid


class _ChannelProj(nn.Module):
    """1×1 conv channel projection used between backbone stages."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.proj(x)))


class AttentionDipoleModel(nn.Module):
    """CNN backbone + attention pooling + DipoleNet heads.

    Architecture (with base_channels=32):
      Input [B, 3, 100, 100]
      -> stem Conv(3→32) + BN + ReLU
      -> stage1: ResBlock(32)×2, MaxPool2d(2)
      -> stage2: Proj(32→64), ResBlock(64)×2, MaxPool2d(2)
      -> stage3: Proj(64→128), ResBlock(128)×2, MaxPool2d(2)
      -> stage4: Proj(128→256), ResBlock(256)×2, MaxPool2d(2)
      -> AttentionPooling(256) -> [B, 256]
      -> DipoleNet(256) -> (moment [B,3], direction [B,3], log_mag [B,1])

    Parameters ≈ 3.2 M, fits comfortably on an RTX 4080 at batch 128.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(
            ResidualBlockStandardBN(c), ResidualBlockStandardBN(c),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            _ChannelProj(c, c * 2),
            ResidualBlockStandardBN(c * 2), ResidualBlockStandardBN(c * 2),
            nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            _ChannelProj(c * 2, c * 4),
            ResidualBlockStandardBN(c * 4), ResidualBlockStandardBN(c * 4),
            nn.MaxPool2d(2),
        )
        self.stage4 = nn.Sequential(
            _ChannelProj(c * 4, c * 8),
            ResidualBlockStandardBN(c * 8), ResidualBlockStandardBN(c * 8),
            nn.MaxPool2d(2),
        )
        backbone_dim = c * 8  # 256
        self.pool = AttentionPooling(backbone_dim)
        self.head = DipoleNet(backbone_dim)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        features = self.pool(x)
        return self.head(features)  # (moment, direction, log_mag)


class DipoleNetLoss(nn.Module):
    """Combined loss for AttentionDipoleModel.

    Separate cosine direction loss and log-space magnitude loss, balanced
    via Kendall homoscedastic uncertainty weighting.  An optional
    physics-consistency term penalises deviation from the analytic dipole
    forward model.

    Args:
        physics_weight: Fixed weight γ for the physics-consistency term.
            Set to 0 to disable.
        eps: Small value for numerical stability.
    """

    def __init__(self, physics_weight: float = 0.1, eps: float = 1e-8) -> None:
        super().__init__()
        self.physics_weight = float(physics_weight)
        self.eps = eps
        self.log_sigma_dir = nn.Parameter(torch.zeros(1))
        self.log_sigma_mag = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        moment: torch.Tensor,
        direction: torch.Tensor,
        log_mag: torch.Tensor,
        target: torch.Tensor,
        input_field: torch.Tensor | None = None,
        r_grid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute combined loss.

        Args:
            moment: Predicted moment vectors [B, 3].
            direction: Predicted unit direction vectors [B, 3].
            log_mag: Predicted log-magnitudes [B, 1].
            target: Ground-truth moment vectors [B, 3].
            input_field: Observed field maps [B, 3, H, W] for physics loss.
            r_grid: Measurement grid [H, W, 3] for physics loss.
        Returns:
            (total_loss, component_dict) where component_dict has keys
            "L_dir", "L_mag", "L_physics".
        """
        target_norm = target.norm(dim=1, keepdim=True).clamp(min=self.eps)
        target_dir = target / target_norm
        target_log_mag = torch.log(target_norm.squeeze(1))

        cos_sim = (direction * target_dir).sum(dim=1).clamp(-1 + self.eps, 1 - self.eps)
        L_dir = (1.0 - cos_sim).mean()

        L_mag = F.mse_loss(log_mag.squeeze(1), target_log_mag)

        inv_sq_dir = torch.exp(-2.0 * self.log_sigma_dir)
        inv_sq_mag = torch.exp(-2.0 * self.log_sigma_mag)
        loss = inv_sq_dir * L_dir + self.log_sigma_dir + inv_sq_mag * L_mag + self.log_sigma_mag

        L_physics = torch.tensor(0.0, device=moment.device)
        if self.physics_weight > 0.0 and input_field is not None and r_grid is not None:
            L_physics = forward_consistency_loss(moment, input_field, r_grid)
            loss = loss + self.physics_weight * L_physics

        return loss, {
            "L_dir": L_dir.item(),
            "L_mag": L_mag.item(),
            "L_physics": L_physics.item(),
        }


# Compute inclination and declination losses separately


class ResidualBlockStandardBN(nn.Module):
    """Residual block variant that uses standard BatchNorm without blend scheduling."""

    def __init__(self, channel_count: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_count, channel_count, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_count)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(channel_count, channel_count, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_count)

    def forward(self, input_features):
        residual_features = input_features
        convolved_features = self.conv1(input_features)
        convolved_features = self.bn1(convolved_features)
        convolved_features = self.activation(convolved_features)
        convolved_features = self.conv2(convolved_features)
        convolved_features = self.bn2(convolved_features)
        return self.activation(convolved_features + residual_features)

