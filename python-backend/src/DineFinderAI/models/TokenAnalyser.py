import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import json


class TokenAnalyser:
  class RestaurantNERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
      self.encodings = encodings

    def __len__(self):
      return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
      return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

  MODEL_PATH = Path.cwd().joinpath("resources/model/")
  NER_OUTPUT = Path.cwd().joinpath("resources/model/training")
  # Label mapping
  LABELS_TO_IDS = {"O": 0, "LOC": 1, "RES": 2, "CUISINE": 3, "AMBIENCE": 4}
  IDS_TO_LABELS = {v: k for k, v in LABELS_TO_IDS.items()}

  def __init__(self, new_tokens: bool = False) -> None:
    self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    self.model = BertForTokenClassification.from_pretrained(
      "bert-base-uncased", num_labels=5
    )  # O, LOC, RES, CUISINE, AMBIENCE
    if new_tokens:
      import json

      token_loc = Path.cwd().joinpath("resources/new_tokens.json")
      with open(token_loc, "r") as file:
        tokens = json.load(file)

      new_tokens_to_add = []
      new_tokens_to_add += tokens["city"]
      new_tokens_to_add += tokens["restaurant_related"]

      self.tokenizer.add_tokens(new_tokens_to_add)
      self.model.resize_token_embeddings(len(self.tokenizer))
      print(f"Tokenizer saved to {self.MODEL_PATH.joinpath("tokenizer_modified")}")
      self.tokenizer.save_pretrained(self.MODEL_PATH.joinpath("tokenizer_modified"))

  def tokenize_and_align_labels(self, data: dict[list[str], list[str]]):
    """
    Tokenize the text and align labels for BERT NER training.
    """
    tokenized_inputs = self.tokenizer(
      [item["tokens"] for item in data], is_split_into_words=True, truncation=True, padding=True
    )
    labels = []
    for i, example in enumerate(data):
      word_ids = tokenized_inputs.word_ids(i)
      label_ids = []
      for word_id in word_ids:
        if word_id is None:
          label_ids.append(-100)  # Ignore special tokens
          continue
        label_ids.append(self.LABELS_TO_IDS[example["labels"][word_id]])
      labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

  def train(self, data: dict[list[str], list[str]]) -> None:
    self.NER_OUTPUT.mkdir(exist_ok=True)
    train_dataset = self.tokenize_and_align_labels(data)
    train_dataset = self.RestaurantNERDataset(train_dataset)

    # K-Fold Cross-Validation
    k = 5  # Number of folds
    kf = KFold(n_splits=k)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_dataset)))):
      print(f"\nFold {fold + 1}/{k}")
      log_path = self.NER_OUTPUT.joinpath("logs")
      log_path.mkdir(exist_ok=True)
      # Create train and validation datasets for this fold
      train_subset = Subset(train_dataset, train_idx)
      val_subset = Subset(train_dataset, val_idx)

      training_args = TrainingArguments(
        output_dir=self.NER_OUTPUT.joinpath(f"fold_{fold}"),
        evaluation_strategy="steps",
        logging_dir=log_path.joinpath(f"fold_{fold}"),
        logging_strategy="epoch",
        per_device_train_batch_size=64,
        num_train_epochs=3,
        save_steps=10,
        logging_steps=10,
        save_total_limit=2,
        learning_rate=5e-5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
      )

      trainer = Trainer(
        model=self.model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=val_subset,
      )

      trainer.train()

      eval_results = trainer.evaluate()
      fold_results.append(eval_results)

    model_path = Path.cwd().joinpath("resources", "model")
    trainer.save_model(model_path.joinpath("NER"))

    avg_loss = sum([result["eval_loss"] for result in fold_results]) / k
    print(f"\nAverage Validation Loss Across {k} Folds: {avg_loss:.4f}")

    log_history = trainer.state.log_history
    with open(log_path.joinpath("NER_training_logs.json"), "w") as f:
      json.dump(log_history, f, indent=4)
