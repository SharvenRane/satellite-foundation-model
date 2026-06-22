"""Self supervised pretraining loop for the masked autoencoder."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .model import MAE


@dataclass
class PretrainResult:
    losses: list[float] = field(default_factory=list)

    @property
    def first(self) -> float:
        return self.losses[0]

    @property
    def last(self) -> float:
        return self.losses[-1]


def pretrain_mae(
    model: MAE,
    images: torch.Tensor,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> PretrainResult:
    """Train the MAE by masked reconstruction and return the loss per epoch.

    The masking pattern is driven by a seeded generator so the run is
    reproducible. Loss is the mean over batches of the masked reconstruction
    error for that epoch.
    """

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = images.shape[0]
    result = PretrainResult()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        epoch_loss = 0.0
        nb = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = images[idx]
            loss, _, _ = model(batch, generator=gen)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            nb += 1
        result.losses.append(epoch_loss / max(1, nb))
    return result


@torch.no_grad()
def extract_features(model: MAE, images: torch.Tensor) -> torch.Tensor:
    """Pooled encoder features for a batch of images, shape (N, dim)."""

    model.eval()
    return model.encoder(images)
