# Goodreads Genre Classification

This project fine-tunes `distilbert-base-cased` to classify Goodreads reviews by book genre. It prepares the Goodreads review data, trains a Hugging Face model, tracks training with Weights & Biases, evaluates the model, and saves the final reports.

## Setup

```bash
python3 -m pip install -r requirements.txt
wandb login
```

For Hugging Face upload, set your token as an environment variable before training:

```bash
export HF_TOKEN=your_huggingface_token
```

Do not commit API keys or tokens in this repository.

## Run

The scripts use `genre_reviews_dict.pickle` by default, so the dataset does not need to be downloaded again.



If `HF_TOKEN` is set, `train.py` will also push the trained model and tokenizer to Hugging Face Hub.

## CPU Note

If training on CPU is too slow, reduce this value in `scripts/data.py`:

```python
REVIEWS_PER_GENRE = 200
```

Then run the three scripts again.

## Results

| Metric | Score |
|-----------|--------|
| Accuracy | 0.58562 |
| F1 Score | 0.58294 |
| Eval Loss | 2.36314 |


- Kaggle Notebook https://www.kaggle.com/code/ashwanig20ait2022/mlops-assignment-2-fine-tuning-classification
- Hugging Face model: https://huggingface.co/Nlp0187/distilbert-goodreads-genres
- W&B dashboard: https://wandb.ai/ashwinmmmec794-iit-jodhpur/mlops-assignment2?nw=nwuserashwinmmmec794

## Outputs

- Prepared data: `prepared_reviews.pkl`
- Fine-tuned model: `distilbert-reviews-genres/`
- Trainer logs: `results/trainer/`, `logs/`
- Evaluation files: `results/eval_metrics.json`, `results/classification_report.json`, `results/predictions.csv`

## Model Choice

I chose `distilbert-base-cased` because it is a compact version of BERT that keeps most of BERT's language understanding ability while being faster and lighter to fine-tune. This is useful for Goodreads genre classification because reviews are natural-language inputs, and a transformer model can use context better than a basic bag-of-words model. DistilBERT is also practical for this assignment because it trains faster on limited GPU resources, making experiment tracking and iteration easier. The cased version preserves capitalization, which can sometimes help with book titles, names, and review text. Overall, it gives a good balance between accuracy, training cost, and reproducibility for a student-scale MLOps workflow.
