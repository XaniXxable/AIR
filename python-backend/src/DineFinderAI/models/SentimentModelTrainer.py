import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class SentimentModelTrainer:
  """
  A class to train and evaluate a sentiment classification model using BERT with K-Fold cross-validation.
  """

  LABELS = {"Positive": 0, "Neutral": 1, "Negative": 2}

  def __init__(
    self,
    model_name: str | Path = "bert-base-uncased",
    max_length: int = 128,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
  ):
    """
    Initializes the SentimentModelTrainer.

    Args:
        model_name (str | Path): Pretrained BERT model name or location.
        max_length (int): Maximum sequence length for tokenization.
        batch_size (int): Batch size for training and validation.
        learning_rate (float): Learning rate for the optimizer.
    """
    self.tokenizer = BertTokenizer.from_pretrained(model_name)
    self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(self.LABELS.items()))
    self.max_length = max_length
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)

  class SentimentDataset(Dataset):
    """
    Custom PyTorch Dataset for tokenized sentiment classification data.
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer: BertTokenizer, max_length: int):
      """
      Initializes the SentimentDataset.

      Args:
          texts (list[str]): List of input text samples.
          labels (list[int]): List of sentiment labels (e.g., 0, 1, 2).
          tokenizer (BertTokenizer): Tokenizer for converting text to BERT-compatible tokens.
          max_length (int): Maximum sequence length for tokenization.
      """
      self.texts = texts
      self.labels = labels
      self.tokenizer = tokenizer
      self.max_length = max_length

    def __len__(self) -> int:
      """Returns the number of samples in the dataset."""
      return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
      """Retrieves a tokenized sample and its label."""
      text = self.texts[idx]
      label = self.labels[idx]
      encoding = self.tokenizer(
        text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
      )
      return {
        "input_ids": encoding["input_ids"].squeeze(),
        "attention_mask": encoding["attention_mask"].squeeze(),
        "label": torch.tensor(label, dtype=torch.long),
      }

  def prepare_data(self, file_path: str) -> pd.DataFrame:
    """
    Loads and preprocesses the dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset with sentiment labels mapped to integers.
    """
    df = pd.read_csv(file_path)
    df["sentiment"] = df["sentiment"].map({"Positive": 0, "Neutral": 1, "Negative": 2})
    return df

  def train_k_fold(
    self, report_path: Path, df: pd.DataFrame, k: int = 5, epochs: int = 3
  ) -> dict[str, dict[str, float]]:
    """
    Performs k-fold cross-validation training and evaluation.

    Args:
        report_path (Path): Path to save the averaged classification report as a JSON file.
        df (pd.DataFrame): The dataset as a DataFrame with 'text' and 'sentiment' columns.
        k (int): Number of folds for cross-validation.
        epochs (int): Number of epochs for training in each fold.

    Returns:
        Dict[str, Dict[str, float]]: Average classification metrics across all folds.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    texts, labels = df["text"], df["sentiment"]

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
      print(f"Starting Fold {fold + 1}/{k}")

      train_texts, val_texts = texts.iloc[train_idx].tolist(), texts.iloc[val_idx].tolist()
      train_labels, val_labels = labels.iloc[train_idx].tolist(), labels.iloc[val_idx].tolist()

      train_dataset = self.SentimentDataset(train_texts, train_labels, self.tokenizer, self.max_length)
      val_dataset = self.SentimentDataset(val_texts, val_labels, self.tokenizer, self.max_length)

      train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
      val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

      model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels).to(self.device)
      optimizer = AdamW(model.parameters(), lr=self.learning_rate)

      for epoch in range(epochs):
        print(f"  Epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc="    Training"):
          optimizer.zero_grad()
          input_ids = batch["input_ids"].to(self.device)
          attention_mask = batch["attention_mask"].to(self.device)
          labels = batch["label"].to(self.device)

          outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
          loss = outputs.loss
          train_loss += loss.item()
          loss.backward()
          optimizer.step()
        print(f"    Training loss: {train_loss / len(train_loader):.4f}")

      model.eval()
      val_loss = 0
      predictions, true_labels = [], []
      with torch.no_grad():
        for batch in tqdm(val_loader, desc="    Validation"):
          input_ids = batch["input_ids"].to(self.device)
          attention_mask = batch["attention_mask"].to(self.device)
          labels = batch["label"].to(self.device)

          outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
          val_loss += outputs.loss.item()
          preds = torch.argmax(outputs.logits, dim=1)
          predictions.extend(preds.cpu().numpy())
          true_labels.extend(labels.cpu().numpy())
      print(f"    Validation loss: {val_loss / len(val_loader):.4f}")

      fold_report = classification_report(
        true_labels, predictions, target_names=["Positive", "Neutral", "Negative"], output_dict=True
      )
      fold_results.append(fold_report)
      print(classification_report(true_labels, predictions, target_names=["Positive", "Neutral", "Negative"]))

    avg_results = self._average_fold_results(fold_results)

    self._save_classification_report(avg_results, report_path)

    return avg_results

  def _average_fold_results(self, fold_results: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """
    Averages the classification metrics across all folds.

    Args:
        fold_results (List[Dict[str, Dict[str, float]]]): List of classification reports for each fold.

    Returns:
        Dict[str, Dict[str, float]]: Averaged classification metrics.
    """
    avg_results = {}
    for key in fold_results[0].keys():
      if isinstance(fold_results[0][key], dict):
        avg_results[key] = {
          metric: sum(fold[key][metric] for fold in fold_results) / len(fold_results) for metric in fold_results[0][key]
        }
      else:
        avg_results[key] = sum(fold[key] for fold in fold_results) / len(fold_results)
    return avg_results

  def _save_classification_report(self, report: dict[str, dict[str, float]], file_path: str) -> None:
    """
    Saves the classification report to a JSON file and prints key metrics.

    Args:
        report (Dict[str, Dict[str, float]]): Classification report containing per-class and overall metrics.
        file_path (str): Path to save the classification report as a JSON file.
    """
    import json

    # Save the report
    with open(file_path, "w") as f:
      json.dump(report, f, indent=4)
    print(f"\nClassification report saved to {file_path}")

    # Display key metrics
    print("\n=== Key Metrics ===")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print("\nPer-Class Metrics:")
    for sentiment, metrics in report.items():
      if sentiment not in ["accuracy", "macro avg", "weighted avg"]:
        print(f"  {sentiment.capitalize()}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1-Score:  {metrics['f1-score']:.4f}")
        print(f"    Support:   {metrics['support']}")


def predict(self, texts: list[str]) -> list[dict[str, float]]:
  """
  Predicts the sentiment of a list of text samples and returns probabilities for each class.

  Args:
      texts (List[str]): List of text samples to predict.

  Returns:
      List[Dict[str, float]]: Predicted sentiment probabilities for each class.
      Each prediction includes:
          - class: The predicted class (e.g., 0, 1, 2).
          - probabilities: A dictionary with class probabilities (e.g., {0: 0.8, 1: 0.15, 2: 0.05}).
  """
  self.model.eval()
  encodings = self.tokenizer(texts, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
  input_ids = encodings["input_ids"].to(self.device)
  attention_mask = encodings["attention_mask"].to(self.device)

  with torch.no_grad():
    outputs = self.model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)  # Compute probabilities

  # Convert predictions to a readable format
  results = []
  for i, probs in enumerate(probabilities):
    class_probs = {class_idx: prob.item() for class_idx, prob in enumerate(probs)}
    predicted_class = torch.argmax(probs).item()
    results.append({"class": predicted_class, "probabilities": class_probs})
  return results
