"""Downstream linear probing of frozen encoder features.

We freeze the pretrained encoder, extract pooled features, and fit a logistic
regression classifier on top. The same probe is fit on features from a randomly
initialized encoder to give an honest baseline. If pretraining did anything
useful, the pretrained features should classify the held out patches better
than the random ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .model import MAE
from .pretrain import extract_features


@dataclass
class ProbeResult:
    train_accuracy: float
    test_accuracy: float


def linear_probe(
    encoder_model: MAE,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    max_iter: int = 500,
    seed: int = 0,
) -> ProbeResult:
    """Fit logistic regression on frozen features and report accuracy."""

    train_feats = extract_features(encoder_model, train_images).cpu().numpy()
    test_feats = extract_features(encoder_model, test_images).cpu().numpy()
    y_train = train_labels.cpu().numpy()
    y_test = test_labels.cpu().numpy()

    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(train_feats, y_train)

    train_acc = float((clf.predict(train_feats) == y_train).mean())
    test_acc = float((clf.predict(test_feats) == y_test).mean())
    return ProbeResult(train_accuracy=train_acc, test_accuracy=test_acc)


def chance_level(labels: torch.Tensor, num_classes: int) -> float:
    """Accuracy of always predicting the most frequent class."""

    counts = np.bincount(labels.cpu().numpy(), minlength=num_classes)
    return float(counts.max() / counts.sum())
