from interface.request.queryRequest import QueryRequest
from interface.response.queryResponse import QueryResponse

from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import json
import torch
import pandas as pd
import sqlite3
import torch.nn.functional as F


class DineFinder:
  resource_folder = Path.cwd().joinpath("resources")

  def __init__(self) -> None:
    self.restaurants = self.load_from_db()
    self.json_data = self.load_json("categ_in_city_queries.json")
    self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    self.model = AutoModel.from_pretrained("bert-base-uncased")

    self.embeddings_folder = self.resource_folder.joinpath("embeddings")
    self.embeddings_folder.mkdir(exist_ok=True)

    self.restaurant_emeddings_location = self.embeddings_folder.joinpath("restaurant_emeddings.save")
    if not self.restaurant_emeddings_location.exists():
      self.save_restaurant_embeddings()
    self.restaurant_embeddings = self.load_restaurant_embeddings(self.restaurant_emeddings_location)

  def save_restaurant_embeddings(self):
    """
    Generate and save embeddings for all restaurants in the DataFrame.
    """
    restaurant_embeddings = []
    for _, row in self.restaurants.iterrows():
      description = (
        f"{row['name']} located at {row['address']} offers {row['categories']}. "
        f"It has {row['stars']} stars and {row['review_count']} reviews."
      )
      embedding = self.get_bert_embedding(description)
      restaurant_embeddings.append(
        {
          "name": row["name"],
          "address": row["address"],
          "categories": row["categories"],
          "stars": row["stars"],
          "review_count": row["review_count"],
          "city": row["city"],
          "embedding": embedding,
        }
      )

    torch.save(restaurant_embeddings, self.restaurant_emeddings_location)
    print(f"Embeddings saved to {self.restaurant_emeddings_location}")

  def load_restaurant_embeddings(self, load_path: str):
    """
    Load pre-generated restaurant embeddings from a file.

    Args:
        load_path (str): Path to the saved embeddings.

    Returns:
        list: List of dictionaries containing restaurant names and their embeddings.
    """
    embeddings = torch.load(load_path)
    print(f"Loaded {len(embeddings)} embeddings from {load_path}")
    return embeddings

  def get_bert_embedding(self, text: str):
    """
    Generate an embedding for the given text using BERT.
    Args:
        text (str): The input text.
    Returns:
        torch.Tensor: The mean-pooled embedding.
    """
    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

    with torch.no_grad():
      outputs = self.model(**inputs)
      hidden_states = outputs.last_hidden_state

    embedding = hidden_states.mean(dim=1)
    return embedding.squeeze()

  def get_top_10_restaurants(self, query: str):
    """
    Find the top 10 restaurants matching a query using BERT embeddings.

    Args:
        query (str): The user query.
        restaurant_df (pd.DataFrame): DataFrame containing restaurant details.

    Returns:
        list: Top 10 restaurants with similarity scores.
    """
    city_list = self.get_unique_cities()

    # Extract the city from the query
    city = self.extract_city_from_query(query, city_list)
    if not city:
      print("City not found in query.")
      return []

    print(f"Extracted City: {city}")

    query_embedding = self.get_bert_embedding(query)

    filtered_embeddings = [
      restaurant for restaurant in self.restaurant_embeddings if restaurant["city"].lower() == city.lower()
    ]

    if not filtered_embeddings:
      print(f"No restaurants found for city: {city}")
      return []

    scores = []
    details = []
    for restaurant in filtered_embeddings:
      score = F.cosine_similarity(query_embedding.unsqueeze(0), restaurant["embedding"].unsqueeze(0)).item()
      scores.append(score)
      details.append(restaurant)

    scores_tensor = torch.tensor(scores)

    top_scores, top_indices = torch.topk(scores_tensor, min(10, len(scores_tensor)))

    top_restaurants = [
      {
        "name": details[idx].get("name", "Unknown"),
        "address": details[idx].get("address", "Address not available"),
        "categories": details[idx].get("categories", "Categories not available"),
        "stars": details[idx].get("stars", "Rating not available"),
        "review_count": details[idx].get("review_count", "Review count not available"),
        "score": top_scores[i].item(),
      }
      for i, idx in enumerate(top_indices)
    ]

    return top_restaurants

  def load_from_db(self) -> pd.DataFrame:
    with sqlite3.connect(self.resource_folder.joinpath("database.db")) as conn:
      # Query the table
      query = "SELECT * FROM restaurants"
      df = pd.read_sql_query(query, conn)
    return df

  def extract_city_from_query(self, query: str, city_list: list) -> str:
    """
    Extract the city from the query using a predefined list of cities.

    Args:
        query (str): The user query.
        city_list (list): List of unique cities from the database.

    Returns:
        str: The extracted city name or None if not found.
    """
    # query_lower = query.lower()
    # matched_cities = [city for city in city_list if city.lower() in query_lower]
    # print(matched_cities)
    # if len(matched_cities) == 1:
    #   return matched_cities[0]
    # elif len(matched_cities) > 1:
    #   raise ValueError(f"Multiple cities found in the query: {matched_cities}. Please specify only one.")

    return "philadelphia"

  def get_unique_cities(self) -> list:
    """
    Extract unique cities from the database.

    Returns:
        list: List of unique city names.
    """
    with sqlite3.connect(self.resource_folder.joinpath("database.db")) as conn:
      query = "SELECT DISTINCT city FROM restaurants"
      result = pd.read_sql_query(query, conn)
    return result["city"].tolist()

  def load_json(self, file_name: str) -> dict[str, any]:
    with open(self.resource_folder.joinpath(file_name), "r") as file:
      content = json.load(file)
    print(f"File {file_name} loaded")
    return content

  def execute(self, query: QueryRequest) -> QueryResponse:
    return self.get_top_10_restaurants(query.UserInput)
