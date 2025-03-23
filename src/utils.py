import numpy as np
from sklearn.metrics import classification_report
from transformers import EvalPrediction
from datasets import load_metric

# Load evaluation metric (e.g., seqeval for NER)
metric = load_metric("seqeval")

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> tuple:
    """
    Align predictions with labels for NER evaluation.
    Args:
        predictions: Model predictions (logits).
        label_ids: Ground truth labels.
    Returns:
        A tuple of aligned predictions and labels.
    """
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape

    aligned_preds = []
    aligned_labels = []

    for i in range(batch_size):
        # Remove padding tokens (label_id = -100)
        mask = label_ids[i] != -100
        aligned_preds.append(preds[i][mask])
        aligned_labels.append(label_ids[i][mask])

    return aligned_preds, aligned_labels

def compute_metrics(p: EvalPrediction) -> dict:
    """
    Compute evaluation metrics for NER.
    Args:
        p: EvalPrediction object containing predictions and labels.
    Returns:
        A dictionary of metrics (precision, recall, F1, accuracy).
    """
    preds, labels = align_predictions(p.predictions, p.label_ids)
    results = metric.compute(predictions=preds, references=labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def preprocess_data(dataset):
    """
    Preprocess the CoNLL-03 dataset for training.
    Args:
        dataset: Hugging Face dataset object.
    Returns:
        Preprocessed dataset.
    """
    def tokenize_and_align_labels(examples):
        tokenized_inputs = model.tokenize(examples["tokens"], labels=examples["ner_tags"])
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    return tokenized_datasets

def save_model(model, path: str):
    """
    Save the trained model to a file.
    Args:
        model: Trained model.
        path: Path to save the model.
    """
    torch.save(model.state_dict(), path)

def load_model(model, path: str):
    """
    Load a trained model from a file.
    Args:
        model: Model architecture.
        path: Path to the saved model.
    """
    model.load_state_dict(torch.load(path))
    return model