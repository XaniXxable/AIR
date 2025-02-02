import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from pathlib import Path
import pandas as pd


class RestaurantNER:
  """
  A class for performing Named Entity Recognition (NER) on restaurant-related data 
  using a fine-tuned BERT model.

  Attributes:
      IDS_TO_LABELS (dict): Mapping of label IDs to their string representations (e.g., "LOC", "RES").
      data (pd.DataFrame): Restaurant data for processing.
      model (BertForTokenClassification): The fine-tuned BERT model for NER.
      tokenizer (BertTokenizerFast): Tokenizer for preprocessing input text.
      device (torch.device): Device to run the model on (CPU or GPU).
  """
  # CUSTOM_TOKENS = ["fast food", "bubble tea"]
  IDS_TO_LABELS = {0: "O", 1: "LOC", 2: "RES", 3: "CUISINE", 4: "AMBIENCE"}

  def __init__(self, model_path: Path, restaurant_data: pd.DataFrame, custom_tokenizer: bool = False) -> None:
    """
    Initializes the RestaurantNER system with a fine-tuned model and tokenizer.

    Args:
        model_path (Path): Path to the fine-tuned BERT model directory.
        restaurant_data (pd.DataFrame): DataFrame containing restaurant-related data.
        custom_tokenizer (bool, optional): Whether to use a custom tokenizer. Defaults to False.
    """
    self.data = restaurant_data
    self.model = BertForTokenClassification.from_pretrained(model_path)

    tokernizer_name = (
      "bert-base-uncased" if not custom_tokenizer else Path.cwd().joinpath("resources/model/tokenizer_modified/")
    )

    self.tokenizer = BertTokenizerFast.from_pretrained(tokernizer_name)

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)
    self.model.eval()  # Set the model to evaluation mode

  def _extract_entities(self, query: str) -> list[tuple[str, str]]:
    """
    Extract entities from a query using the fine-tuned BERT model.

    Args:
        query (str): The input query (e.g., "Can you recommend an Italian restaurant in New York?").

    Returns:
        list: A list of tuples (word, label) for extracted entities.
    """
    tokens = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)

    with torch.no_grad():
      outputs = self.model(**tokens)

    predicted_ids = torch.argmax(outputs.logits, dim=2).squeeze().tolist()

    predicted_labels = [self.IDS_TO_LABELS[label_id] for label_id in predicted_ids]

    words = self.tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze())

    entities = [(word, label) for word, label in zip(words, predicted_labels) if label != "O"]
    return entities

  def __call__(self, query: str) -> pd.DataFrame:
    return self.filter_restaurants(query)

  def filter_restaurants(self, query: str) -> pd.DataFrame:
    """
    Filters restaurant data based on the extracted entities (location and cuisine) from the query.

    Args:
        query (str): The input query string.

    Returns:
        pd.DataFrame: A pandas DataFrame of restaurants matching the location and cuisine in the query.
    """
    entities = self._extract_entities(query)

    location = (
      " ".join([word for word, label in entities if label == "LOC"])
      .replace("[SEP]", "")
      .replace("[CLS]", "")
      .replace("?", "")
      .strip()
    )
    cuisines = [word for word, label in entities if label == "CUISINE"]

    if not location:
      print("No location detected in the query.")
      return pd.DataFrame()

    filtered_restaurants = self.data[self.data["city"].str.lower() == location.lower()]

    if filtered_restaurants.empty:
      print(f"No restaurants found in {location}.")
      return filtered_restaurants

    # If cuisine is not provided, take the most common cuisine
    if not cuisines:
      most_common_cuisine = filtered_restaurants["categories"].value_counts().idxmax()  # Most frequent cuisine
      print(f"No cuisine specified. Using most common cuisine: {most_common_cuisine}")
      cuisines.append(most_common_cuisine)

    return self._filter_by_cuisines(filtered_restaurants, cuisines)

  def _filter_by_cuisines(self, data: pd.DataFrame, cuisines: list[str]) -> pd.DataFrame:
    """
    Filters the input DataFrame to include only rows with specified cuisines.

    Args:
        data (pd.DataFrame): The input DataFrame containing restaurant data. 
                             Must include "categories" and "name" columns.
        cuisines (list[str]): A list of cuisine names to filter by.

    Returns:
        pd.DataFrame: A DataFrame containing rows where the "categories" column matches 
                      any of the specified cuisines, with duplicate restaurant names removed.
    """
    final_filtered_df = pd.DataFrame()
    final_filtered_list = []

    for cuisine in cuisines:
      for _, row in data.iterrows():
        if cuisine.lower() in row["categories"].lower() and row["name"] not in [r["name"] for r in final_filtered_list]:
          final_filtered_list.append(row)

    final_filtered_df = pd.DataFrame(final_filtered_list)
    return final_filtered_df


def load_from_db() -> pd.DataFrame:
  """
  Loads restaurant data from the database into a pandas DataFrame.

  Returns:
      pd.DataFrame: A DataFrame containing the restaurant data from the database.
  """
  import sqlite3

  db_loc = Path.cwd().joinpath("resources", "database.db")
  with sqlite3.connect(db_loc) as conn:
    query = "SELECT * FROM restaurants"
    df = pd.read_sql_query(query, conn)
  return df


if __name__ == "__main__":
  model_path = Path.cwd().joinpath("resources/model/NER/")
  data = load_from_db()
  ner_system = RestaurantNER(model_path, data)

  query = "Recommend a good fast food restaurant in New Orleans"
  filtered_restaurants = ner_system.filter_restaurants(query)

  print(f"Filtered restaurants for '{query}':")
  print(filtered_restaurants)
