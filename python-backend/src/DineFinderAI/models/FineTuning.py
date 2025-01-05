from DineFinderAI.models.DineFinder import DineFinder
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW
import json
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import sqlite3
import pandas as pd
import logging

# Set logging level to ERROR to suppress large print messages
logging.getLogger("transformers").setLevel(logging.ERROR)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class QueryResponseScoringModel(nn.Module):
  def __init__(self, model_name="bert-base-uncased"):
    super(QueryResponseScoringModel, self).__init__()
    self.bert_mlm = BertForMaskedLM.from_pretrained(model_name)
    self.scoring_head = nn.Linear(768 * 2, 1)  # Linear layer for score prediction

  def forward(self, query_input_ids, query_attention_mask, response_input_ids, response_attention_mask):
    # Get CLS embeddings for query and response
    query_outputs = self.bert_mlm.bert(
      input_ids=query_input_ids, attention_mask=query_attention_mask
    ).last_hidden_state[:, 0, :]
    response_outputs = self.bert_mlm.bert(
      input_ids=response_input_ids, attention_mask=response_attention_mask
    ).last_hidden_state[:, 0, :]

    # Concatenate the CLS embeddings
    combined = torch.cat((query_outputs, response_outputs), dim=1)

    # Pass through scoring head and apply sigmoid to get score between 0 and 1
    score = torch.sigmoid(self.scoring_head(combined))

    return score.squeeze()


class FineTuning(DineFinder):
  """
  This class is responsible to fine-tune the base model.
  """

  def __init__(self, data_file, tokenizer, max_length=128):
    with open(data_file, "r") as f:
      data = json.load(f)
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.pairs = []

    # Create pairs of (query, response, score)
    for query, response_data in data.items():
      for response, score in zip(response_data["responses"], response_data["scores"]):
        self.pairs.append((query, response, score))

  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, idx):
    query, response, score = self.pairs[idx]

    # Tokenize query and response
    query_inputs = self.tokenizer(
      query, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
    )
    response_inputs = self.tokenizer(
      response, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
    )

    return {
      "query_input_ids": query_inputs["input_ids"].squeeze(),
      "query_attention_mask": query_inputs["attention_mask"].squeeze(),
      "response_input_ids": response_inputs["input_ids"].squeeze(),
      "response_attention_mask": response_inputs["attention_mask"].squeeze(),
      "score": torch.tensor(score, dtype=torch.float),
    }


def main() -> None:
  training_data = Path.cwd().joinpath("resources", "best_places_by_city_queries.json")
  model_folder = Path.cwd().joinpath("resources", "model")
  criterion = nn.MSELoss()  # Loss for regression task
  model_folder.mkdir(exist_ok=True)

  model_save_loc = model_folder.joinpath("trained.save")
  token_save_loc = model_folder.joinpath("token.save")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = QueryResponseScoringModel("bert-base-uncased").to(device)

  dataset = FineTuning(training_data, tokenizer)
  dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

  # Optimizer
  optimizer = AdamW(model.parameters(), lr=5e-5)

  # Training loop
  model.to(device)

  epochs = 3
  for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
      # Move inputs to device
      query_input_ids = batch["query_input_ids"].to(device)
      query_attention_mask = batch["query_attention_mask"].to(device)
      response_input_ids = batch["response_input_ids"].to(device)
      response_attention_mask = batch["response_attention_mask"].to(device)
      scores = batch["score"].to(device)  # True relevance scores

      # Forward pass
      predicted_scores = model(query_input_ids, query_attention_mask, response_input_ids, response_attention_mask)

      # Compute loss
      loss = criterion(predicted_scores, scores)
      total_loss += loss.item()

      # Backpropagation and optimization
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

  # Save the fine-tuned model
  torch.save(model.state_dict(), model_save_loc)
  tokenizer.save_pretrained(token_save_loc)


def load_from_db() -> pd.DataFrame:
  db_loc = Path.cwd().joinpath("resources", "database.db")
  with sqlite3.connect(db_loc) as conn:
    query = "SELECT * FROM restaurants"
    df = pd.read_sql_query(query, conn)
  return df


def precompute_restaurant_embeddings(max_length=128):
  """
  Precompute embeddings for a list of restaurants and save both embeddings and texts.
  """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model_folder = Path.cwd().joinpath("resources", "model")
  model_save_loc = model_folder.joinpath("trained.save")
  token_save_loc = model_folder.joinpath("token.save")
  save_path = model_folder.joinpath("trained_embeddings.save")

  tokenizer = BertTokenizer.from_pretrained(model_folder)
  model = BertForMaskedLM.from_pretrained(model_folder).to(device)

  model.eval()
  data = load_from_db()

  # Check for the 'categories' column
  if "categories" not in data.columns:
    raise ValueError("The dataset must contain a 'categories' column with restaurant descriptions.")

  embeddings = []
  restaurant_details = []
  for _, restaurant in data.iterrows():
    description = f"{restaurant['name']} located at {restaurant['address']}, {restaurant['city']}, {restaurant['state']} offering {restaurant['categories']}."
    restaurant_details.append(
      {
        "name": restaurant["name"],
        "address": restaurant["address"],
        "city": restaurant["city"],
        "country": restaurant["state"],
        "categories": restaurant["categories"],
      }
    )
    inputs = tokenizer(
      description, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length"
    ).to(device)
    with torch.no_grad():
      outputs = model.bert(inputs["input_ids"], attention_mask=inputs["attention_mask"])
      cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
    embeddings.append(cls_embedding.squeeze(0).cpu())  # Store CPU tensor for saving

  # Save restaurant embeddings and texts together
  embeddings = torch.stack(embeddings)
  torch.save({"embeddings": embeddings, "details": restaurant_details}, save_path)
  print(f"Saved embeddings and texts to {save_path}")


def get_top_responses_for_query(query):
  """
  Get the top-ranked restaurants for a query using precomputed embeddings.

  Args:
      query (str): User's input query.
      embedding_file (str): Path to saved embeddings and restaurant texts.

  Returns:
      list: Top 10 restaurants with relevance scores.
  """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model_folder = Path.cwd().joinpath("resources", "model")
  save_path = model_folder.joinpath("trained_embeddings.save")

  tokenizer = BertTokenizer.from_pretrained(model_folder)
  model = BertForMaskedLM.from_pretrained(model_folder).to(device)

  # Load precomputed restaurant data
  restaurant_data = torch.load(save_path)
  print(f"Loaded {len(restaurant_data)} embeddings from {save_path}")

  restaurant_embeddings = restaurant_data["embeddings"].to(device)
  restaurant_details = restaurant_data["details"]
  city = "philadelphia"
  filtered_indices = [i for i, r in enumerate(restaurant_details) if r["city"].lower() == city.lower()]
  filtered_embeddings = restaurant_embeddings[filtered_indices]
  filtered_details = [restaurant_details[i] for i in filtered_indices]

  query_inputs = tokenizer(query, return_tensors="pt", max_length=128, truncation=True, padding="max_length").to(device)
  with torch.no_grad():
    query_outputs = model.bert(query_inputs["input_ids"], attention_mask=query_inputs["attention_mask"])
    query_embedding = query_outputs.last_hidden_state[:, 0, :]  # CLS token embedding
    query_embedding = F.normalize(query_embedding, p=2, dim=1)  # Normalize for cosine similarity

  # Normalize filtered restaurant embeddings and compute cosine similarity
  filtered_embeddings = F.normalize(filtered_embeddings, p=2, dim=1)
  similarities = torch.matmul(query_embedding, filtered_embeddings.T).squeeze()

  # Number of available restaurants in the city
  num_restaurants = filtered_embeddings.size(0)
  top_k = min(10, num_restaurants)  # Get top 10 (or fewer if fewer are available)

  # Get indices of top-k results
  top_indices = similarities.topk(top_k).indices

  # Retrieve top-ranked restaurant details
  top_restaurants = [filtered_details[idx] for idx in top_indices.tolist()]
  top_scores = [similarities[idx].item() for idx in top_indices.tolist()]

  # Combine details and scores
  return [
    {
      "name": r["name"],
      "address": r["address"],
      "city": r["city"],
      "country": r["country"],
      "categories": r["categories"],
      "score": score,
    }
    for r, score in zip(top_restaurants, top_scores)
  ]
