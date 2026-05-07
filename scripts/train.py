"""Load DistilBERT, set up Trainer, and fine-tune the classifier."""

from __future__ import annotations

import inspect
import os
import pickle
from pathlib import Path

try:
    from .utils import ReviewDataset, compute_metrics, set_seed
except ImportError:
    from utils import ReviewDataset, compute_metrics, set_seed


PREPARED_DATA_PATH = "data/prepared_reviews.pkl"
MODEL_NAME = "distilbert-base-cased"
MODEL_OUTPUT_DIR = "distilbert-reviews-genres"
TRAINING_OUTPUT_DIR = "results/trainer"
LOGGING_DIR = "logs"
NUM_TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 3e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
SEED = 42
WANDB_PROJECT = "mlops-assignment2"
WANDB_RUN_NAME = "distilbert-run-1"
HF_REPO_ID = "Nlp0187/distilbert-goodreads-genres"


def load_prepared_data(path: str | Path) -> dict:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def build_training_args():
    """Build TrainingArguments across transformers versions."""
    from transformers import TrainingArguments

    training_kwargs = {
        "output_dir": TRAINING_OUTPUT_DIR,
        "logging_dir": LOGGING_DIR,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "logging_steps": LOGGING_STEPS,
        "save_strategy": SAVE_STRATEGY,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "report_to": "wandb",
        "run_name": WANDB_RUN_NAME,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        training_kwargs["eval_strategy"] = EVAL_STRATEGY
    else:
        training_kwargs["evaluation_strategy"] = EVAL_STRATEGY

    return TrainingArguments(**training_kwargs)


def train() -> Trainer:
    import wandb
    from huggingface_hub import login
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
    from transformers import Trainer

    set_seed(SEED)

    data = load_prepared_data(PREPARED_DATA_PATH)
    label2id = data["label2id"]
    id2label = {int(key): value for key, value in data["id2label"].items()}

    train_dataset = ReviewDataset(data["train_encodings"], data["train_labels_encoded"])
    test_dataset = ReviewDataset(data["test_encodings"], data["test_labels_encoded"])

    model = DistilBertForSequenceClassification.from_pretrained(
        data.get("model_name", MODEL_NAME),
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
    training_args = build_training_args()

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model": data.get("model_name", MODEL_NAME),
            "epochs": NUM_TRAIN_EPOCHS,
            "batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_length": data.get("max_length"),
            "dataset": "UCSD Goodreads",
            "train_examples": len(train_dataset),
            "test_examples": len(test_dataset),
        },
    )

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(MODEL_OUTPUT_DIR)

        tokenizer = DistilBertTokenizerFast.from_pretrained(data.get("model_name", MODEL_NAME))
        tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            trainer.model.push_to_hub(HF_REPO_ID, private=False)
            tokenizer.push_to_hub(HF_REPO_ID, private=False)
            hf_model_url = f"https://huggingface.co/{HF_REPO_ID}"
            wandb.run.summary["huggingface_model"] = hf_model_url
            print(f"Pushed model and tokenizer to {hf_model_url}")
        else:
            print("HF_TOKEN not set. Saved locally but skipped Hugging Face upload.")

        print(f"Saved fine-tuned model to {MODEL_OUTPUT_DIR}")
        return trainer
    finally:
        wandb.finish()


if __name__ == "__main__":
    train()
