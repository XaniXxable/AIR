from pathlib import Path
from typing import Any
from DineFinderAI.db.DatabaseManager import DatabaseManager
import pandas as pd
import json
import re
import random

def is_city_separated(city_name: str) -> bool:
  """
  Check if a city is separated with some separators.

  Args:
    city_name (str): name to check

  Returns:
    boolean: Ture if it is separated otherwise False
  """
  separators = [" ", "_", "-"]
  return any(sep in city_name for sep in separators)


def is_category_separated(category: str) -> bool:
  """
  Check if a category is separated with some separators.

  Args:
    city_name (str): name to check

  Returns:
    boolean: True if it is separated otherwise False
  """
  separators = [" ", "_", "-"]
  return any(sep in category for sep in separators)


def generate_category_queries_and_tokens_for_all_cities(df: pd.DataFrame, output_file: Path) -> None:
  """
  Tokenize category-city queries and write the result to a json file.

  Args:
      df (pd.DataFrame): A DataFrame containing at least the columns `city` and `categories`.
      output_file (Path): The file path where the generated JSON output will be saved.
  """
  # Normalize city and category columns to ensure case-insensitivity
  df["city_normalized"] = df["city"].str.lower()
  df["categories_normalized"] = df["categories"].str.lower().str.split(", ")

  # Create all unique city-category combinations
  city_category_combinations = pd.MultiIndex.from_product(
    [df["city_normalized"].unique(), df["categories_normalized"].explode().unique()], names=["city", "category"]
  ).to_frame(index=False)
  results = []
  found_tokens = {"city": [], "cuisines": []}

  for _, row in city_category_combinations.iterrows():
    query = "Where can I find good CATEGORY in CITY"
    # query = f"Where can I find good {row["category"]} in {row["city"]}"
    tmp = {"tokens": [], "labels": []}

    if is_city_separated(row["city"]):
      found_tokens["city"].append(row["city"])

    if is_category_separated(row["category"]):
      found_tokens["cuisines"].append(row["category"])

    for string in query.split(" "):
      string = string.replace("CATEGORY", row["category"])
      string = string.replace("CITY", row["city"])
      pattern_category = rf'\b{row["category"]}\b'
      pattern_city = rf'\b{row["city"]}\b'
      # if string not in row["category"] and string not in row["city"]:
      if not re.search(pattern_category, string) and not re.search(pattern_city, string):
        tmp["tokens"].append(string)
        tmp["labels"].append("O")
        continue

      if string in row["category"]:
        tmp["tokens"].append(string)
        tmp["labels"].append("CUISINE")
      elif string in row["city"]:
        tmp["tokens"].append(string)
        tmp["labels"].append("LOC")
    results.append(tmp)
  write_to_json(results, output_file)


# Integrate the new cities and cuisines into the dataset generator
def generate_enhanced_dataset(new_cities, new_cuisines, num_samples=4000) -> list[dict[str, str]]:
    locations = [
        "New York", "San Francisco", "Chicago", "Austin",
        "Tokyo", "Los Angeles", "Paris", "Cape Town",
        "Rome", "Rio de Janeiro", "Bangkok", "Berlin"
    ] + new_cities  # Adding new cities
    
    cuisines = [
        "Italian", "Mexican", "Sushi", "Chinese",
        "Bubble Tea", "Ice Cream", "Vegan", "Indian Street Food",
        "Thai", "Ramen", "Korean BBQ", "Peruvian", "Tex-Mex", "Seafood"
    ] + new_cuisines  # Adding new cuisines

    templates = [
        ("Find {} restaurants in {}.", ["CUISINE", "O", "O", "LOC"]),
        ("Where can I get {} food in {}?", ["O", "O", "O", "CUISINE", "O", "LOC"]),
        ("Best places for {} near {}.", ["O", "O", "O", "CUISINE", "O", "LOC"]),
        ("I’m craving {} in {}.", ["O", "O", "CUISINE", "O", "LOC"]),
        ("Show me {} spots in {}.", ["O", "O", "CUISINE", "O", "LOC"]),
        ("Any {} food nearby?", ["O", "CUISINE", "O", "O"]),
        ("Looking for cheap {} in {}.", ["O", "O", "CUISINE", "O", "LOC"])
    ]

    dataset = []
    for _ in range(num_samples):
        cuisine = random.choice(cuisines)
        location = random.choice(locations)
        template, labels_template = random.choice(templates)
        
        # Fill template with cuisine and location
        query = template.format(cuisine, location)
        
        # Ensure multi-word entities stay as one token
        tokens = query.replace("?", "").replace(".", "").split()
        query_tokens = []
        labels = []
        for token in tokens:
            if cuisine in query and token in cuisine.split():
                if not query_tokens or query_tokens[-1] != cuisine:
                    query_tokens.append(cuisine)
                    labels.append("CUISINE")
            elif location in query and token in location.split():
                if not query_tokens or query_tokens[-1] != location:
                    query_tokens.append(location)
                    labels.append("LOC")
            else:
                query_tokens.append(token)
                labels.append("O")

        dataset.append({"tokens": query_tokens, "labels": labels})

    return dataset


def write_to_json(data: dict[str, Any], json_file: Path) -> None:
  """
  Write query and response data to a JSON file.

  Args:
      data (dict[str, any]): The dict containing query and response columns.
      json_file (str): Path to the JSON file to write.
  """
  with open(json_file, "a") as file:
    file.write(json.dumps(data, indent=2))
  print(f"Saved {json_file.resolve()}")


def generate_enhanced() -> None:
  print("Generate extended query-token dataset...")
  resources_path = Path.cwd().joinpath("resources")
  
  file_path = Path.cwd().joinpath("resources", "new_tokens.json")
  with open(file_path, 'r') as f:
      additional_data = json.load(f)

  # Extract cities and cuisines
  new_cities = additional_data["city"]
  new_cuisines = additional_data["restaurant_related"] + additional_data["unrelated_cuisines"]
  
  enhanced_dataset = generate_enhanced_dataset(new_cities, new_cuisines)

  enhanced_file_path = resources_path.joinpath("ner_training_dataset_enhanced.json")
  write_to_json(enhanced_dataset, enhanced_file_path)

def generate_basic() -> None:
  print("Generate basic query-token dataset...")
  resources_path = Path.cwd().joinpath("resources")
  db_path = resources_path.joinpath("database.db")
  db_manager = DatabaseManager(database_filepath=db_path)

  db_manager.connectFunc()
  data_frame = db_manager.execute("SELECT name, address, city, state, stars, categories from restaurants")
  db_manager.closeFunc()
  output_file = resources_path.joinpath("query_learning.json")
  generate_category_queries_and_tokens_for_all_cities(data_frame, output_file)


if __name__ == "__main__":
  import sys

  sys.exit(generate_basic())
