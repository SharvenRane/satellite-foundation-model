import torch

from src.data import make_unlabeled
from src.model import MAE, MAEEncoder, PatchEmbed


def test_patch_embed_tilify_roundtrip():
    embed = PatchEmbed(bands=6, img_size=32, tile=8, dim=64)
    x = torch.randn(3, 6, 32, 32)
    tokens = embed.tilify(x)
    assert tokens.shape == (3, 16, 6 * 8 * 8)
    recon = embed.untilify(tokens)
    # tilify then untilify must be a lossless rearrangement.
    assert torch.allclose(recon, x, atol=1e-5)


def test_encoder_feature_shape():
    enc = MAEEncoder(bands=6, img_size=32, tile=8, dim=64, depth=2, heads=4)
    x = torch.randn(5, 6, 32, 32)
    feats = enc(x)
    assert feats.shape == (5, 64)


def test_masking_keeps_and_restores_consistently():
    model = MAE(bands=6, img_size=32, tile=8, mask_ratio=0.6)
    gen = torch.Generator().manual_seed(0)
    ids_keep, ids_restore, mask = model.random_masking(4, generator=gen)
    n = model.num_tokens
    # mask is binary and hides exactly the non kept tokens.
    assert mask.shape == (4, n)
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}
    kept = (mask == 0).sum(dim=1)
    assert torch.all(kept == ids_keep.shape[1])
    # ids_restore is a valid permutation per row.
    for row in ids_restore:
        assert sorted(row.tolist()) == list(range(n))


def test_forward_returns_scalar_loss_and_masked_pred():
    model = MAE(bands=6, img_size=32, tile=8, mask_ratio=0.6)
    x = make_unlabeled(n=6, bands=6, size=32, seed=2)
    gen = torch.Generator().manual_seed(1)
    loss, pred, mask = model(x, generator=gen)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0
    assert pred.shape == (6, model.num_tokens, model.encoder.embed.tile_pixels)


def test_gradients_flow_to_encoder():
    model = MAE(bands=6, img_size=32, tile=8, mask_ratio=0.6)
    x = make_unlabeled(n=4, bands=6, size=32, seed=5)
    gen = torch.Generator().manual_seed(0)
    loss, _, _ = model(x, generator=gen)
    loss.backward()
    grads = [
        p.grad for p in model.encoder.parameters() if p.requires_grad
    ]
    assert any(g is not None and torch.any(g != 0) for g in grads)
