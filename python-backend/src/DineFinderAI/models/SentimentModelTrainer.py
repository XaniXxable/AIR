import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
from pathlib import Path


from transformers import Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd


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
    self.model_name = model_name
    self.num_labels = len(self.LABELS.items())
    self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=self.num_labels)
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
          text,
          max_length=self.max_length,
          padding="max_length",
          truncation=True,
          return_tensors="pt"
      )
      return {
          "input_ids": encoding["input_ids"].squeeze(0),
          "attention_mask": encoding["attention_mask"].squeeze(0),
          "label": torch.tensor(label, dtype=torch.long)
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

  # Metrics Function
  def compute_metrics(self, pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

  def train_k_fold(
    self, report_path, df, epochs=3, k=5
  ):
    """
    Trains a sentiment classification model using k-fold cross-validation.

    Args:
        df (pd.DataFrame): DataFrame with 'text' and 'sentiment' columns.
        tokenizer: Pretrained tokenizer for text processing.
        max_length (int): Maximum sequence length for tokenization.
        batch_size (int): Batch size for training and validation.
        learning_rate (float): Learning rate for the optimizer.
        num_labels (int): Number of output classes.
        model_name (str): Pretrained BERT model name.
        device: Torch device (e.g., 'cuda' or 'cpu').
        epochs (int): Number of epochs for training.
        k (int): Number of folds for cross-validation.

    Returns:
        dict: Averaged classification metrics across all folds.
    """
    # Ensure texts and labels are pandas Series
    texts = df["text"].tolist()
    labels = df["sentiment"].map({"Positive": 0, "Neutral": 1, "Negative": 2}).tolist()

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"=== Fold {fold + 1}/{k} ===")
        
        # Split data into train and validation sets
        train_texts, val_texts = np.array(texts)[train_idx], np.array(texts)[val_idx]
        train_labels, val_labels = np.array(labels)[train_idx], np.array(labels)[val_idx]

        # Create datasets
        train_dataset = self.SentimentDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = self.SentimentDataset(val_texts, val_labels, self.tokenizer, self.max_length)

        # Load model
        model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels).to(self.device)

        # Define TrainingArguments
        training_args = TrainingArguments(
            output_dir=Path.cwd().joinpath("resources", f"model_fold_{fold + 1}"),
            evaluation_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir=f"logs_fold_{fold + 1}",
            logging_steps=50,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        # Train model
        trainer.train()

        # Evaluate model
        eval_results = trainer.evaluate()
        fold_results.append(eval_results)

        # Save model
        trainer.save_model(Path.cwd().joinpath("resources", f"model_fold_{fold + 1}"),)
        print(f"Model for fold {fold + 1} saved.")

    # Average results across folds
    avg_results = {
        metric: np.mean([result[metric] for result in fold_results]) for metric in fold_results[0]
    }
    print("\n=== Average Results Across Folds ===")
    print(avg_results)
    
    self.model.save_pretrained(Path.cwd().joinpath("resources", "model"))
    # self._save_classification_report(avg_results, report_path)

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


  def predict(self, texts: list[str]) -> list[dict[str, int] | dict[str, dict[int, float]]]:
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

def process_row(args):
  reviews_df, trainer = args  # Unpack the tuple
  # business_id = row["business_id"]
  # name = row["name"]
  # filtered_df = reviews_df[reviews_df["business_id"] == business_id]
  review_texts = reviews_df["text"].to_list()
  results = trainer.predict(review_texts)

  positive = [entry["probabilities"][0] for entry in results]
  neutral = [entry["probabilities"][1] for entry in results]
  negative = [entry["probabilities"][2] for entry in results]
  positive_avg = sum(positive) / len(results)
  neutral_avg = sum(neutral) / len(results)
  negative_avg = sum(negative) / len(results)
  
  return {
      # "business_id": business_id,
      # "name": name,
      "positive_avg": positive_avg,
      "neutral_avg": neutral_avg,
      "negative_avg": negative_avg
  }

def predict_review_class_for_restaurant(model_path: Path):
  from DineFinderAI.db.DatabaseManager import DatabaseManager
  from multiprocessing import set_start_method, Pool
  trainer = SentimentModelTrainer(model_path)
  reviews_file_path = Path.cwd().joinpath("resources", "yelp_academic_dataset_review.json")
  db_manager = DatabaseManager(Path.cwd().joinpath("resources", "database.db"))
  db_manager.connectFunc()
  business_id_df = db_manager.execute("SELECT business_id, name FROM restaurants")
  chunks = []
  chunk_size = 100000

  print("Read review file...")
  with pd.read_json(reviews_file_path, orient="records", lines=True, chunksize=chunk_size) as reader:
    for chunk in reader:
      # Process the chunk (e.g., filter or aggregate) -> avoid memory issues
      chunks.append(chunk[["business_id", "text"]])

  reviews_df = pd.concat(chunks, ignore_index=True)

  print("Start with predicting the class of the reviews for each restaurant...")
  
  # set_start_method("spawn")
  # # Create a list of arguments for each row
  # rows_with_args = [(reviews_df[reviews_df["business_id"] == row["business_id"]], trainer) for _, row in business_id_df.iterrows()]

  # # Use multiprocess Pool for parallel processing
  # with Pool() as pool:
  #   results = list(tqdm(pool.imap(process_row, rows_with_args), total=len(rows_with_args)))

  for _, row in tqdm(business_id_df.iterrows(), total=len(business_id_df)):
    business_id = row["business_id"]
    # name = row["name"]
    filtered_df = reviews_df[reviews_df["business_id"] == business_id]
    reviews_df = reviews_df[reviews_df["business_id"] != business_id]
    review_texts = filtered_df["text"].to_list()[0:5]
    results = trainer.predict(review_texts)

    positive = [entry["probabilities"][0] for entry in results]
    neutral = [entry["probabilities"][1] for entry in results]
    negative = [entry["probabilities"][2] for entry in results]
    positive_avg = sum(positive) / len(results)
    neutral_avg = sum(neutral) / len(results)
    negative_avg = sum(negative) / len(results)

    # print(f"AVG results for {name} -> Positive: {positive_avg}, Neutral: {neutral_avg}, Negative: {negative_avg}")

  db_manager.closeFunc()

def train():
  resources_path = Path.cwd().joinpath("resources")
  report_path = resources_path.joinpath("report.json")
  model_path = resources_path.joinpath("model")
  if not model_path.is_dir():
    trainer = SentimentModelTrainer()
    df = pd.read_csv(resources_path.joinpath("restaurant_reviews_sample.csv"))
    trainer.train_k_fold(report_path, df)
  
  predict_review_class_for_restaurant(model_path)

if __name__ == "__main__":
  import sys

  sys.exit(train())
