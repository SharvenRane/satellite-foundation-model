"""Synthetic multispectral satellite patch generation.

The generator builds small multiband image patches that carry real structure:
smooth low frequency backgrounds, oriented texture, and per band offsets that
stand in for the spectral signature of different surfaces. There is no random
noise only output, so a self supervised encoder has actual signal to model and
a downstream classifier has a learnable boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PatchConfig:
    bands: int = 6
    size: int = 32
    seed: int = 0


# Amplitude of the clean class signal, its spatial frequency, and the standard
# deviation of the class independent sensor noise. The noise is several times
# larger than the signal so that the recoverable structure, not the raw pixels,
# is what carries the label.
SIGNAL_AMP: float = 0.6
SIGNAL_FREQ: float = 0.5
NOISE_STD: float = 1.4


def _radial_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return yy, xx


def make_patch(
    rng: np.random.Generator,
    bands: int,
    size: int,
    class_id: int,
    num_classes: int,
) -> np.ndarray:
    """Build one multispectral patch whose class lives in its spatial texture.

    Each class is defined by the orientation and spatial frequency of an
    oriented sinusoidal texture, the multispectral analogue of distinct land
    cover types (cropland rows, ripple patterns, urban grid, and so on). The
    class signal is deliberately kept out of the simple first order statistics:
    every patch is normalized per band to zero mean and unit variance, the per
    band spectral offset is randomized independently of the class, and the
    texture phase and background are random. As a result the discriminative
    information sits in spatial frequency content, which a randomly initialized
    encoder pools away but a model that has learned spatial structure can read.
    """

    yy, xx = _radial_grid(size)

    # Class dependent low frequency structure: a smooth oriented ramp whose
    # orientation is set by the class. This is the clean signal a denoising
    # model can recover, the stand in for a coherent land cover gradient. Its
    # amplitude is deliberately small relative to the noise added below.
    angle = math.pi * class_id / max(1, num_classes)
    direction = np.cos(angle) * xx + np.sin(angle) * yy
    phase = rng.uniform(0.0, 0.4 * math.pi)
    signal = SIGNAL_AMP * np.sin(
        2.0 * math.pi * SIGNAL_FREQ * direction + phase
    ).astype(np.float32)

    # Per band spectral offset is random and independent of the class, so it
    # carries no label information and cannot be exploited by simple pooling.
    spectral = rng.normal(0.0, 1.0, size=bands).astype(np.float32)

    patch = np.empty((bands, size, size), dtype=np.float32)
    for b in range(bands):
        # Strong, high frequency, class independent sensor noise. It dominates
        # the raw pixel variance, so a random encoder that simply projects the
        # pixels is swamped by noise. A masked autoencoder, trained to predict a
        # tile from its neighbours, has to average this noise away and represent
        # the coherent low frequency signal instead, which is what makes its
        # features more useful downstream.
        noise = rng.normal(0.0, NOISE_STD, size=(size, size)).astype(np.float32)
        patch[b] = signal + spectral[b] + noise

    # Normalize each band to zero mean, unit variance. This strips the per band
    # offset and overall energy, forcing any classifier to rely on the spatial
    # pattern rather than first order statistics.
    patch = patch - patch.mean(axis=(1, 2), keepdims=True)
    std = patch.std(axis=(1, 2), keepdims=True)
    patch = patch / np.clip(std, 1e-3, None)
    return patch.astype(np.float32)


def make_dataset(
    n: int,
    num_classes: int = 4,
    bands: int = 6,
    size: int = 32,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (images, labels) with images shaped (n, bands, size, size)."""

    rng = np.random.default_rng(seed)
    images = np.empty((n, bands, size, size), dtype=np.float32)
    labels = np.empty((n,), dtype=np.int64)
    for i in range(n):
        cls = int(rng.integers(0, num_classes))
        images[i] = make_patch(rng, bands, size, cls, num_classes)
        labels[i] = cls
    return torch.from_numpy(images), torch.from_numpy(labels)


def make_unlabeled(
    n: int,
    num_classes: int = 4,
    bands: int = 6,
    size: int = 32,
    seed: int = 0,
) -> torch.Tensor:
    """Patches for pretraining. Labels are not returned."""

    images, _ = make_dataset(
        n, num_classes=num_classes, bands=bands, size=size, seed=seed
    )
    return images
