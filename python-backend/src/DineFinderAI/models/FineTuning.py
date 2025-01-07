from DineFinderAI.models.DineFinder import DineFinder
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, AdamW, BertForSequenceClassification
from pathlib import Path
import logging
import torch.nn.functional as F

# Set logging level to ERROR to suppress large print messages
logging.getLogger("transformers").setLevel(logging.ERROR)


class CustomLossWithScoreConsistency(torch.nn.Module):
  def __init__(self, margin=1.0, lambda_loc=0.7, lambda_score=0.5):
    super(CustomLossWithScoreConsistency, self).__init__()
    self.margin_loss = torch.nn.TripletMarginLoss(margin=margin, p=2)
    self.lambda_loc = lambda_loc  # Controls weight of location penalty
    self.lambda_score = lambda_score  # Controls weight of score consistency loss

  def forward(
    self, anchor_scores, positive_scores, negative_scores, location_mismatch_penalty, predicted_scores, expected_scores
  ):
    # Triplet margin loss for relevance ranking
    triplet_loss = self.margin_loss(anchor_scores, positive_scores, negative_scores)

    # Mean Squared Error (MSE) for score consistency
    mse_loss = F.mse_loss(predicted_scores, expected_scores)

    # Total loss: Triplet loss + location penalty + score consistency loss
    total_loss = triplet_loss + self.lambda_loc * location_mismatch_penalty.mean() + self.lambda_score * mse_loss
    return total_loss


class FineTuning(DineFinder):
  """
  This class is responsible to fine-tune the base model.
  """

  def __init__(self, model_path="bert-base-uncased", device=None):
    self.resource_folder = Path.cwd().joinpath("resources")
    self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=1).to(self.device)
    self.tokenizer = BertTokenizer.from_pretrained(model_path)

  class QueryResponseDataset(Dataset):
    def __init__(self, data):
      """
      Initializes the dataset with query-response pairs and relevance scores.

      Args:
            data (list): List of dictionaries where each dictionary contains:
                - "query": The input query string.
                - "expected_location": The expected city/location (string).
                - "top_10_expected_responses": List of dictionaries with:
                    - "name": Restaurant name.
                    - "address": Full address.
                    - "relevance_score": Relevance score (0 to 1).
      """
      self.samples = []

      for item in data:
        query = item["query"]
        expected_location = item["location"]
        responses = item["responses"]

        for response in responses:
          name = response["name"]
          address = response["address"]
          relevance_score = response["score"]

          # Add sample with query, expected location, and score
          self.samples.append(
            {
              "query": query,
              "response": f"{name} at {address}",
              "score": relevance_score,
              "expected_location": expected_location,
            }
          )

    def __len__(self):
      return len(self.samples)

    def __getitem__(self, idx):
      sample = self.samples[idx]
      return {
        "query": sample["query"],
        "response": sample["response"],
        "score": torch.tensor(sample["score"], dtype=torch.float32),
        "expected_location": sample["expected_location"],
      }

  def get_dataloader(self, dataset, batch_size=8):
    """Returns a DataLoader for batching the dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

  def train(self, data, epochs=3, batch_size=16, lr=2e-5):
    """
    Fine-tunes BERT for query-response scoring using the dataset.

    Args:
        data (dict): Dataset containing query-response pairs and scores.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
    """
    dataset = self.QueryResponseDataset(data)
    dataloader = self.get_dataloader(dataset, batch_size=batch_size)

    optimizer = AdamW(self.model.parameters(), lr=lr)
    loss_fn = CustomLossWithScoreConsistency(margin=1.0, lambda_loc=0.7, lambda_score=0.5)

    self.model.train()

    for epoch in range(epochs):
      total_loss = 0

      for batch in dataloader:
        queries = batch["query"]
        expected_responses = batch["response"]
        scores = batch["score"].to(self.device)
        expected_locations = batch["expected_location"]

        # Tokenize input for model
        input_data = self.tokenizer(
          queries,  # List of queries
          expected_responses,  # Corresponding list of responses
          return_tensors="pt",
          padding=True,
          truncation=True,
          max_length=512,
        ).to(self.device)

        # Forward pass
        outputs = self.model(**input_data)
        logits = outputs.logits  # Raw scores
        predicted_scores = torch.sigmoid(logits.squeeze())  # Normalize scores

        # **Get Top-10 Predicted Responses**
        _, top_indices = torch.topk(predicted_scores, k=10)  # Get top-10 indices
        top_predicted_responses = [expected_responses[i] for i in top_indices]  # Simulate predictions
        print(top_predicted_responses)
        # **Extract Locations from Top-10 Predictions**
        predicted_locations = [resp["expected_location"] for resp in top_predicted_responses]  # Extract cities
        location_match = torch.tensor(
          [
            1 if pred.lower() == exp.lower() else 0
            for pred, exp in zip(predicted_locations, [expected_locations[0]] * 10)
          ],
          dtype=torch.float32,
          device=self.device,
        )

        # Relevance Weighted Location Penalty:
        location_penalty = (1 - location_match) * scores[top_indices]  # Penalize more for highly relevant mismatches
        mismatch_penalty = location_penalty.sum() / 10  # Average penalty for top-10

        # **Expected Scores for Top-10 Predictions**:
        top_expected_scores = scores[top_indices]

        # Triplet Loss + Location Penalty + Score Consistency
        loss = loss_fn(
          predicted_scores[top_indices],
          scores[top_indices],
          torch.zeros_like(scores[top_indices]),
          mismatch_penalty,
          predicted_scores[top_indices],
          top_expected_scores,
        )

        # **Backward Pass and Optimization**
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

      avg_loss = total_loss / len(dataloader)
      print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    self.save_model()

  def save_model(self):
    """
    Saves the fine-tuned model and tokenizer.
    """
    save_path = self.resource_folder.joinpath("model")
    save_path.mkdir(exist_ok=True)
    self.model.save_pretrained(save_path)
    self.tokenizer.save_pretrained(save_path)
    print(f"Model saved to '{save_path}'")
