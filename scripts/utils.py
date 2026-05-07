"""Shared helpers for Goodreads genre classification."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed: int) -> None:
    """Make sampling and model training more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_label_maps(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Create stable label-to-id and id-to-label mappings."""
    unique_labels = sorted(set(labels))
    label2id = {label: index for index, label in enumerate(unique_labels)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


class ReviewDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapper for tokenized review texts and labels."""

    def __init__(self, encodings: dict[str, Any], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def compute_metrics(pred: Any) -> dict[str, float]:
    """Metrics callback used by HuggingFace Trainer."""
    labels = pred.label_ids
    predictions = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def save_json(data: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
