import copy

import torch

from src.data import make_dataset, make_unlabeled
from src.model import MAE
from src.probe import chance_level, linear_probe
from src.pretrain import pretrain_mae


def _build():
    num_classes = 4
    pre_images = make_unlabeled(
        n=256, num_classes=num_classes, bands=6, size=32, seed=0
    )
    tr_img, tr_lab = make_dataset(
        n=200, num_classes=num_classes, bands=6, size=32, seed=10
    )
    te_img, te_lab = make_dataset(
        n=120, num_classes=num_classes, bands=6, size=32, seed=99
    )
    return num_classes, pre_images, tr_img, tr_lab, te_img, te_lab


def test_probe_beats_chance_and_random_features():
    torch.manual_seed(0)
    num_classes, pre_images, tr_img, tr_lab, te_img, te_lab = _build()

    random_model = MAE(bands=6, img_size=32, tile=8, mask_ratio=0.6)
    trained_model = copy.deepcopy(random_model)

    pretrain_mae(trained_model, pre_images, epochs=40, lr=1e-3, seed=0)

    random_res = linear_probe(
        random_model, tr_img, tr_lab, te_img, te_lab, seed=0
    )
    trained_res = linear_probe(
        trained_model, tr_img, tr_lab, te_img, te_lab, seed=0
    )

    chance = chance_level(te_lab, num_classes)

    # Pretrained features must clear chance by a real margin.
    assert trained_res.test_accuracy > chance + 0.15
    # And they must beat the random initialized encoder's features.
    assert trained_res.test_accuracy > random_res.test_accuracy


def test_probe_result_fields_are_valid_probabilities():
    _, pre_images, tr_img, tr_lab, te_img, te_lab = _build()
    model = MAE(bands=6, img_size=32, tile=8)
    res = linear_probe(model, tr_img, tr_lab, te_img, te_lab, seed=0)
    assert 0.0 <= res.train_accuracy <= 1.0
    assert 0.0 <= res.test_accuracy <= 1.0
