import torch

from src.data import make_dataset, make_patch, make_unlabeled
import numpy as np


def test_dataset_shapes_and_dtype():
    images, labels = make_dataset(n=20, num_classes=4, bands=6, size=32, seed=1)
    assert images.shape == (20, 6, 32, 32)
    assert labels.shape == (20,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64
    assert int(labels.min()) >= 0
    assert int(labels.max()) <= 3


def test_dataset_is_reproducible():
    a_img, a_lab = make_dataset(n=10, seed=7)
    b_img, b_lab = make_dataset(n=10, seed=7)
    assert torch.equal(a_lab, b_lab)
    assert torch.allclose(a_img, b_img)


def test_classes_are_distinguishable():
    # Different class ids must produce different patch statistics, otherwise
    # the downstream task would be unsolvable.
    rng = np.random.default_rng(0)
    p0 = make_patch(rng, bands=6, size=32, class_id=0, num_classes=4)
    rng = np.random.default_rng(0)
    p3 = make_patch(rng, bands=6, size=32, class_id=3, num_classes=4)
    assert not np.allclose(p0, p3)


def test_unlabeled_returns_images_only():
    x = make_unlabeled(n=8, bands=6, size=32, seed=3)
    assert x.shape == (8, 6, 32, 32)
