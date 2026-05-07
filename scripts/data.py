"""Load, sample, split, and encode Goodreads review data."""

from __future__ import annotations

import gzip
import json
import pickle
import random
from pathlib import Path

import requests

try:
    from .utils import build_label_maps, set_seed
except ImportError:
    from utils import build_label_maps, set_seed


RAW_DATA_PATH = "data/genre_reviews_dict.pickle"
PREPARED_DATA_PATH = "data/prepared_reviews.pkl"
MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
HEAD = 10000
DOWNLOAD_SAMPLE_SIZE = 2000
REVIEWS_PER_GENRE = 1000
TRAIN_RATIO = 0.8
SEED = 42

GENRE_URLS = {
    "poetry": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}


def load_reviews_from_url(url: str, head: int | None, sample_size: int) -> list[str]:
    """Stream compressed JSONL reviews from a URL and return a random sample."""
    reviews: list[str] = []
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with gzip.open(response.raw, "rt", encoding="utf-8") as file:
            for count, line in enumerate(file, start=1):
                review = json.loads(line).get("review_text")
                if review:
                    reviews.append(review)
                if head is not None and count >= head:
                    break
    return random.sample(reviews, min(sample_size, len(reviews)))


def load_or_download_reviews(
    raw_data_path: Path,
    head: int,
    download_sample_size: int,
    force_download: bool,
) -> dict[str, list[str]]:
    """Load a cached genre-review dictionary or download one."""
    if raw_data_path.exists() and not force_download:
        with raw_data_path.open("rb") as file:
            return pickle.load(file)

    genre_reviews: dict[str, list[str]] = {}
    for genre, url in GENRE_URLS.items():
        print(f"Loading reviews for genre: {genre}")
        genre_reviews[genre] = load_reviews_from_url(url, head, download_sample_size)

    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_data_path.open("wb") as file:
        pickle.dump(genre_reviews, file)
    return genre_reviews


def split_reviews(
    genre_reviews: dict[str, list[str]],
    reviews_per_genre: int,
    train_ratio: float,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Create train/test text and label lists from sampled genre reviews."""
    train_texts: list[str] = []
    train_labels: list[str] = []
    test_texts: list[str] = []
    test_labels: list[str] = []

    for genre, reviews in genre_reviews.items():
        sampled_reviews = random.sample(reviews, min(reviews_per_genre, len(reviews)))
        train_count = int(len(sampled_reviews) * train_ratio)

        train_texts.extend(sampled_reviews[:train_count])
        train_labels.extend([genre] * train_count)
        test_texts.extend(sampled_reviews[train_count:])
        test_labels.extend([genre] * (len(sampled_reviews) - train_count))

    return train_texts, train_labels, test_texts, test_labels


def prepare_data() -> dict[str, object]:
    """Prepare tokenized data and label maps for training/evaluation."""
    from transformers import DistilBertTokenizerFast

    set_seed(SEED)
    raw_data_path = Path(RAW_DATA_PATH)
    output_path = Path(PREPARED_DATA_PATH)

    genre_reviews = load_or_download_reviews(
        raw_data_path=raw_data_path,
        head=HEAD,
        download_sample_size=DOWNLOAD_SAMPLE_SIZE,
        force_download=False,
    )
    train_texts, train_labels, test_texts, test_labels = split_reviews(
        genre_reviews,
        reviews_per_genre=REVIEWS_PER_GENRE,
        train_ratio=TRAIN_RATIO,
    )

    label2id, id2label = build_label_maps(train_labels)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    prepared = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "train_texts": train_texts,
        "train_labels": train_labels,
        "test_texts": test_texts,
        "test_labels": test_labels,
        "label2id": label2id,
        "id2label": id2label,
        "train_encodings": dict(train_encodings),
        "test_encodings": dict(test_encodings),
        "train_labels_encoded": [label2id[label] for label in train_labels],
        "test_labels_encoded": [label2id[label] for label in test_labels],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(prepared, file)

    print(f"Saved prepared data to {output_path}")
    print(f"Train examples: {len(train_texts)} | Test examples: {len(test_texts)}")
    return prepared


if __name__ == "__main__":
    prepare_data()
