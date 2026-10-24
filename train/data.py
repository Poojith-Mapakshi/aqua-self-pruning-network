"""Deterministic synthetic three-class spiral dataset, generated with NumPy only."""

from __future__ import annotations

import numpy as np


def make_spirals(
    samples_per_class: int = 500,
    noise: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return X with shape (N, 2) and integer labels y with shape (N,).

    Each class forms an interleaving spiral. The dataset is generated from
    scratch so no external dataset loader is needed.
    """
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    if noise < 0:
        raise ValueError("noise must be non-negative")

    rng = np.random.default_rng(seed)
    total = samples_per_class * 3
    X = np.zeros((total, 2), dtype=np.float64)
    y = np.repeat(np.arange(3, dtype=np.int64), samples_per_class)

    for class_id in range(3):
        start = class_id * samples_per_class
        end = start + samples_per_class
        radius = np.linspace(0.05, 1.0, samples_per_class)
        angle = np.linspace(0.0, 4.0 * np.pi, samples_per_class)
        angle = angle + class_id * (2.0 * np.pi / 3.0)
        angle = angle + rng.normal(0.0, noise, samples_per_class)
        X[start:end, 0] = radius * np.sin(angle)
        X[start:end, 1] = radius * np.cos(angle)

    # Shuffle deterministically so batches are not grouped by class.
    permutation = rng.permutation(total)
    return X[permutation], y[permutation]


def main() -> None:
    X, y = make_spirals()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Classes: {np.unique(y).tolist()}")
    print(f"First five X rows:\n{X[:5]}")
    print(f"First five labels: {y[:5].tolist()}")


if __name__ == "__main__":
    main()
