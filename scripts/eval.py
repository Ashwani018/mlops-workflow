"""Evaluate a fine-tuned Goodreads genre classifier and save results."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

from sklearn.metrics import classification_report

try:
    from .utils import ReviewDataset, compute_metrics, save_json
except ImportError:
    from utils import ReviewDataset, compute_metrics, save_json


PREPARED_DATA_PATH = "data/prepared_reviews.pkl"
MODEL_OUTPUT_DIR = "distilbert-reviews-genres"
RESULTS_DIR = "results"
WANDB_PROJECT = "mlops-assignment2"
WANDB_RUN_NAME = "distilbert-run-1"


def load_prepared_data(path: str | Path) -> dict:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def save_predictions(
    path: str | Path,
    texts: list[str],
    true_labels: list[str],
    predicted_labels: list[str],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["true_label", "predicted_label", "review_text"],
        )
        writer.writeheader()
        for true_label, predicted_label, text in zip(true_labels, predicted_labels, texts):
            writer.writerow(
                {
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "review_text": text,
                }
            )


def evaluate() -> dict:
    import wandb
    from transformers import DistilBertForSequenceClassification, Trainer

    data = load_prepared_data(PREPARED_DATA_PATH)
    id2label = {int(key): value for key, value in data["id2label"].items()}
    test_dataset = ReviewDataset(data["test_encodings"], data["test_labels_encoded"])

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_OUTPUT_DIR)
    trainer = Trainer(model=model, compute_metrics=compute_metrics)

    wandb.init(
        project=WANDB_PROJECT,
        name=f"{WANDB_RUN_NAME}-eval",
        job_type="evaluation",
        config={
            "model_dir": MODEL_OUTPUT_DIR,
            "data_path": PREPARED_DATA_PATH,
            "dataset": "UCSD Goodreads",
        },
    )

    try:
        metrics = trainer.evaluate(test_dataset)
        prediction_output = trainer.predict(test_dataset)
        predicted_ids = prediction_output.predictions.argmax(-1).flatten().tolist()
        predicted_labels = [id2label[predicted_id] for predicted_id in predicted_ids]

        report = classification_report(
            data["test_labels"],
            predicted_labels,
            output_dict=True,
            zero_division=0,
        )
        text_report = classification_report(
            data["test_labels"],
            predicted_labels,
            zero_division=0,
        )

        metrics_output = f"{RESULTS_DIR}/eval_metrics.json"
        report_output = f"{RESULTS_DIR}/classification_report.json"
        predictions_output = f"{RESULTS_DIR}/predictions.csv"

        save_json(metrics, metrics_output)
        save_json(report, report_output)
        save_predictions(
            predictions_output,
            data["test_texts"],
            data["test_labels"],
            predicted_labels,
        )

        wandb.log(
            {
                "final/loss": metrics.get("eval_loss"),
                "final/accuracy": metrics.get("eval_accuracy"),
                "final/f1": metrics.get("eval_f1"),
            }
        )
        artifact = wandb.Artifact("eval-report", type="evaluation")
        artifact.add_file(report_output)
        wandb.log_artifact(artifact)

        print(metrics)
        print(text_report)
        print(f"Saved metrics to {metrics_output}")
        print(f"Saved classification report to {report_output}")
        print(f"Saved predictions to {predictions_output}")
        return metrics
    finally:
        wandb.finish()


if __name__ == "__main__":
    evaluate()
