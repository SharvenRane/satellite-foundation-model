"""End to end demo: pretrain the encoder, then probe it against a baseline.

Run it with:

    python -m src.demo

It prints the masked reconstruction loss at the start and end of pretraining,
then the linear probe accuracy of the pretrained encoder next to a randomly
initialized encoder and the chance level. Everything is synthetic and runs on
CPU in a few seconds.
"""

from __future__ import annotations

import copy

import torch

from .data import make_dataset, make_unlabeled
from .model import MAE
from .pretrain import pretrain_mae
from .probe import chance_level, linear_probe


def main() -> None:
    torch.manual_seed(0)
    num_classes = 4

    pretrain_images = make_unlabeled(n=256, num_classes=num_classes, seed=0)
    train_images, train_labels = make_dataset(
        n=200, num_classes=num_classes, seed=10
    )
    test_images, test_labels = make_dataset(
        n=120, num_classes=num_classes, seed=99
    )

    random_encoder = MAE(bands=6, img_size=32, tile=8, mask_ratio=0.6)
    pretrained_encoder = copy.deepcopy(random_encoder)

    result = pretrain_mae(
        pretrained_encoder, pretrain_images, epochs=60, lr=1e-3, seed=0
    )
    print("masked reconstruction loss")
    print(f"  first epoch: {result.first:.4f}")
    print(f"  last epoch:  {result.last:.4f}")

    random_probe = linear_probe(
        random_encoder, train_images, train_labels, test_images, test_labels
    )
    trained_probe = linear_probe(
        pretrained_encoder, train_images, train_labels, test_images, test_labels
    )
    chance = chance_level(test_labels, num_classes)

    print()
    print("linear probe test accuracy")
    print(f"  chance level:        {chance:.3f}")
    print(f"  random encoder:      {random_probe.test_accuracy:.3f}")
    print(f"  pretrained encoder:  {trained_probe.test_accuracy:.3f}")


if __name__ == "__main__":
    main()
