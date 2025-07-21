from __future__ import annotations

import os
import numpy as np

from .config import Config


def train_probe(cfg: Config):
    if not os.path.exists(cfg.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {cfg.dataset_path}. Generate data first with data_generation.generate_and_save_dataset()."
        )

    data = np.load(cfg.dataset_path)
    X: np.ndarray = data["X"]  # (N, d_model)
    Y: np.ndarray = data["Y"]  # (N, 2)

    print(f"Loaded dataset X shape {X.shape}, Y shape {Y.shape}")

    # Ordinary least squares with ℓ2 regularization (ridge)
    d_model = X.shape[1]
    lambda_reg = cfg.regularization

    # Compute X^T X and X^T Y
    XTX = X.T @ X  # (d, d)
    XTY = X.T @ Y  # (d, 2)

    if lambda_reg > 0:
        XTX += lambda_reg * np.eye(d_model, dtype=XTX.dtype)

    W = np.linalg.solve(XTX, XTY)  # (d, 2)

    os.makedirs(os.path.dirname(cfg.probe_weights_path), exist_ok=True)
    np.save(cfg.probe_weights_path, W)
    print(f"Saved probe weights to {cfg.probe_weights_path}. Shape: {W.shape}")

    # Simple train MSE
    pred = X @ W  # (N, 2)
    mse = ((pred - Y) ** 2).mean()
    print(f"Train MSE: {mse:.4f}") 