import torch

from src.data import make_unlabeled
from src.model import MAE
from src.pretrain import extract_features, pretrain_mae


def test_pretraining_loss_decreases():
    torch.manual_seed(0)
    images = make_unlabeled(n=96, num_classes=4, bands=6, size=32, seed=0)
    model = MAE(bands=6, img_size=32, tile=8, mask_ratio=0.6)
    result = pretrain_mae(
        model, images, epochs=30, batch_size=32, lr=1e-3, seed=0
    )
    # The masked reconstruction loss at the end of training must be clearly
    # below where it started.
    assert result.last < result.first
    assert result.last < 0.85 * result.first
    assert all(l == l for l in result.losses)  # no NaNs


def test_pretraining_is_reproducible():
    images = make_unlabeled(n=64, seed=0)
    torch.manual_seed(123)
    m1 = MAE(bands=6, img_size=32, tile=8)
    torch.manual_seed(123)
    m2 = MAE(bands=6, img_size=32, tile=8)
    r1 = pretrain_mae(m1, images, epochs=5, seed=123)
    r2 = pretrain_mae(m2, images, epochs=5, seed=123)
    assert torch.allclose(
        torch.tensor(r1.losses), torch.tensor(r2.losses), atol=1e-5
    )


def test_extract_features_shape():
    images = make_unlabeled(n=10, seed=1)
    model = MAE(bands=6, img_size=32, tile=8)
    feats = extract_features(model, images)
    assert feats.shape == (10, model.encoder.dim)
