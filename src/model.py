"""Patch based masked autoencoder for multispectral imagery.

The encoder splits each multiband patch into a grid of small tiles, embeds each
tile, adds learned positional embeddings, and runs a stack of transformer
blocks. During pretraining a random subset of tiles is masked and a lightweight
decoder reconstructs the missing pixels. The same encoder, run on the full set
of tiles, produces the pooled features used for downstream linear probing.

This is a real (if small) MAE style architecture. It is sized so the unit tests
run on CPU in seconds, but nothing about the design is a stub.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _sincos_position(num_tokens: int, dim: int) -> torch.Tensor:
    """Fixed sinusoidal positional embedding, shape (num_tokens, dim)."""

    if dim % 2 != 0:
        raise ValueError("position dim must be even")
    position = torch.arange(num_tokens, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(num_tokens, dim)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Split (B, C, H, W) into non overlapping tiles and linearly embed them."""

    def __init__(self, bands: int, img_size: int, tile: int, dim: int) -> None:
        super().__init__()
        if img_size % tile != 0:
            raise ValueError("img_size must be divisible by tile")
        self.bands = bands
        self.img_size = img_size
        self.tile = tile
        self.grid = img_size // tile
        self.num_tokens = self.grid * self.grid
        self.tile_pixels = bands * tile * tile
        self.proj = nn.Linear(self.tile_pixels, dim)

    def tilify(self, x: torch.Tensor) -> torch.Tensor:
        """Return tokens of raw pixels, shape (B, num_tokens, tile_pixels)."""

        b, c, h, w = x.shape
        t = self.tile
        x = x.reshape(b, c, self.grid, t, self.grid, t)
        # (B, gh, gw, C, t, t)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(b, self.num_tokens, c * t * t)
        return x

    def untilify(self, tokens: torch.Tensor) -> torch.Tensor:
        """Inverse of tilify: (B, num_tokens, tile_pixels) to (B, C, H, W)."""

        b = tokens.shape[0]
        t = self.tile
        c = self.bands
        x = tokens.reshape(b, self.grid, self.grid, c, t, t)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(b, c, self.img_size, self.img_size)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.tilify(x))


class MAEEncoder(nn.Module):
    def __init__(
        self,
        bands: int = 6,
        img_size: int = 32,
        tile: int = 8,
        dim: int = 64,
        depth: int = 3,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.embed = PatchEmbed(bands, img_size, tile, dim)
        self.dim = dim
        pe = _sincos_position(self.embed.num_tokens, dim)
        self.register_buffer("pos_embed", pe, persistent=False)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward_tokens(
        self, tokens: torch.Tensor, ids_keep: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run blocks over embedded tokens with positional embedding added.

        If ids_keep is given (B, k), only those token positions are kept, which
        is how the MAE encoder sees only the visible (unmasked) tiles.
        """

        x = tokens + self.pos_embed.unsqueeze(0)
        if ids_keep is not None:
            idx = ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = torch.gather(x, dim=1, index=idx)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full image to pooled feature vector, shape (B, dim)."""

        tokens = self.embed(x)
        x = self.forward_tokens(tokens, ids_keep=None)
        return x.mean(dim=1)


class MAEDecoder(nn.Module):
    """Small decoder that reconstructs pixels for every tile position."""

    def __init__(
        self,
        num_tokens: int,
        enc_dim: int,
        dim: int,
        tile_pixels: int,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Linear(enc_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)
        pe = _sincos_position(num_tokens, dim)
        self.register_buffer("pos_embed", pe, persistent=False)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, tile_pixels)

    def forward(
        self, latent: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """Scatter visible latents back, fill the rest with the mask token.

        latent: (B, k, enc_dim) encoder output for visible tokens.
        ids_restore: (B, num_tokens) permutation that restores original order.
        Returns predicted pixels per tile, shape (B, num_tokens, tile_pixels).
        """

        b, k, _ = latent.shape
        x = self.proj(latent)
        n_mask = self.num_tokens - k
        mask_tokens = self.mask_token.expand(b, n_mask, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        idx = ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = torch.gather(x, dim=1, index=idx)
        x = x + self.pos_embed.unsqueeze(0)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x)


class MAE(nn.Module):
    """Full masked autoencoder wrapping encoder, decoder, and masking."""

    def __init__(
        self,
        bands: int = 6,
        img_size: int = 32,
        tile: int = 8,
        enc_dim: int = 64,
        enc_depth: int = 3,
        enc_heads: int = 4,
        dec_dim: int = 48,
        dec_depth: int = 2,
        dec_heads: int = 4,
        mask_ratio: float = 0.6,
    ) -> None:
        super().__init__()
        self.encoder = MAEEncoder(
            bands, img_size, tile, enc_dim, enc_depth, enc_heads
        )
        self.mask_ratio = mask_ratio
        embed = self.encoder.embed
        self.decoder = MAEDecoder(
            embed.num_tokens,
            enc_dim,
            dec_dim,
            embed.tile_pixels,
            dec_depth,
            dec_heads,
        )

    @property
    def num_tokens(self) -> int:
        return self.encoder.embed.num_tokens

    def random_masking(
        self, batch: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce keep ids, restore ids, and a binary mask per sample.

        mask is 1 for tokens that are hidden from the encoder (to be predicted),
        0 for visible tokens. This matches the convention used in the loss.
        """

        n = self.num_tokens
        keep = int(round(n * (1.0 - self.mask_ratio)))
        keep = max(1, min(n - 1, keep))
        noise = torch.rand(batch, n, generator=generator)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        mask = torch.ones(batch, n)
        mask[:, :keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return ids_keep, ids_restore, mask

    def forward(
        self, x: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (loss, pred_pixels, mask).

        Loss is mean squared error computed only over masked tiles, which is the
        standard masked autoencoding objective.
        """

        tokens = self.encoder.embed(x)
        target = self.encoder.embed.tilify(x)
        b = x.shape[0]
        ids_keep, ids_restore, mask = self.random_masking(b, generator)
        ids_keep = ids_keep.to(x.device)
        ids_restore = ids_restore.to(x.device)
        mask = mask.to(x.device)

        latent = self.encoder.forward_tokens(tokens, ids_keep=ids_keep)
        pred = self.decoder(latent, ids_restore)

        per_token = ((pred - target) ** 2).mean(dim=-1)
        denom = mask.sum()
        loss = (per_token * mask).sum() / torch.clamp(denom, min=1.0)
        return loss, pred, mask


class LinearProbe(nn.Module):
    """A single linear layer on top of frozen encoder features."""

    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(feats)
