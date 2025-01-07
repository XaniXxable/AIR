import yaml
from pathlib import Path
from typing import Any
from DineFinderAI.db.DatabaseManager import DatabaseManager
import pandas as pd
import json
import re


def process_restaurant_data(df: pd.DataFrame) -> pd.DataFrame:
  """
  Process the DataFrame to generate queries and responses.

  Args:
      df (pd.DataFrame): The raw DataFrame from the database.

  Returns:
      pd.DataFrame: A DataFrame with query and response columns added.
  """

  df["query"] = "What are the details for " + df["name"] + "?"

  df["response"] = (
    df["name"]
    + " is located at "
    + df["address"]
    + " in "
    + df["city"]
    + ", "
    + df["state"]
    + ". It has "
    + df["stars"].astype(str)
    + " stars and is famous in categories like "
    + df["categories"]
    + "."
  )
  return df


def filter_rows_by_string(dataframe, column_name, search_string, case_sensitive=False) -> pd.DataFrame:
  """
  Filters rows in a DataFrame where a specific string appears in a given column.

  Parameters:
      dataframe (pd.DataFrame): The DataFrame to filter.
      column_name (str): The name of the column to search.
      search_string (str): The string to search for.
      case_sensitive (bool): Whether the search should be case-sensitive. Default is False.

  Returns:
      pd.DataFrame: A DataFrame containing only the rows where the string appears.
  """
  if column_name not in dataframe.columns:
    raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

  # Apply the filter using str.contains
  filtered_df = dataframe[dataframe[column_name].str.contains(search_string, case=case_sensitive, na=False)]

  return filtered_df


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


def is_category_separated(category: str):
  """
  Check if a category is separated with some separators.

  Args:
    city_name (str): name to check

  Returns:
  boolean: Ture if it is separated otherwise False
  """
  separators = [" ", "_", "-"]
  return any(sep in category for sep in separators)


def generate_category_queries_for_all_cities(
  df: pd.DataFrame, output_file: Path, checkpoint_file: Path, batch_size: int = 100
):
  """
  Process 'Where can I find good {category} in {city}?' queries with scores,
  saving progress in intervals to a YAML file and using a checkpoint file.

  Args:
      df (pd.DataFrame): DataFrame containing restaurant details.
      output_file (Path): JSON file path to save the results.
      checkpoint_file (Path): File path to save the checkpoint (last processed index).
      batch_size (int): Number of rows to process before saving progress.
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
  # found_tokens["city"] = list((set(found_tokens["city"])))
  # found_tokens["cuisines"] = list(set(found_tokens["cuisines"]))
  # write_to_json(found_tokens, Path.cwd().joinpath("resources/new_tokens.json"))


def count_matching_categories(target_categories: str, search_categories: str) -> float:
  matches = 0
  targets = target_categories.split(",")
  searches = search_categories.split(", ")
  for target_category in targets:
    for search_category in searches:
      if target_category.lower().strip() == search_category.lower().strip():
        matches += 1

  return matches / len(targets)


def generate_best_places_by_city(db_manager: DatabaseManager, output_file: Path) -> None:
  """
  Process 'What are the best restaurants in {city}?' queries with scores.

  Args:
      db_manager (DatabaseManager): The DatabaseManager containing restaurant details.
      output_file (Path): JSON file path to save the results.
  """
  df = db_manager.execute("SELECT DISTINCT(city) FROM restaurants")
  results = []
  for _, row in df.iterrows():
    city = row["city"]
    restaurants = f'SELECT name, address, stars, review_count FROM restaurants WHERE city = "{city}"'
    restaurants_df = db_manager.execute(restaurants)
    highest_review_count = restaurants_df.sort_values(by="review_count", ascending=False).iloc[0]["review_count"]

    # normalize + scoring for the responses
    restaurants_df["normalized_rating"] = restaurants_df["stars"] / 5.0
    restaurants_df["normalized_review_count"] = restaurants_df["review_count"] / highest_review_count
    restaurants_df["score"] = (
      0.75 * restaurants_df["normalized_rating"] + 0.25 * restaurants_df["normalized_review_count"]
    )

    sorted_restaurants = restaurants_df.sort_values(by="score", ascending=False)

    # NOTE: If you want to round the score, uncomment the following line
    sorted_restaurants["score"] = sorted_restaurants["score"].round(3)

    # Generate query and responses for found matches
    query = f"What are the best restaurants in {city}?"
    responses = []
    for _, restaurant in sorted_restaurants.iterrows():
      # response = (
      #   f"One of the best restaurants in {city} is {restaurant['name']} "
      #   + f"at {restaurant['address']}, with {restaurant['stars']} stars "
      #   + f"and {restaurant['review_count']} reviews."
      # )
      response = {
        "name": restaurant["name"],
        "address": restaurant["address"],
        "score": restaurant["score"],
      }
      responses.append(response)
    responses = sorted(responses, key=lambda x: x["score"], reverse=True)[:10]
    results.append({"query": query, "location": city, "responses": responses})
  write_to_json(results, output_file)


def write_to_yaml(df: pd.DataFrame, yaml_file: Path) -> None:
  """
  Write query and response data to a YAML file.

  Args:
      df (pd.DataFrame): The DataFrame containing query and response columns.
      yaml_file (str): Path to the YAML file to write.
  """
  data: list[dict[str, Any]] = df[["query", "response"]].to_dict(orient="records")  # type: ignore

  with open(yaml_file, "w") as file:
    yaml.dump(data, file, default_flow_style=False)

  print(f"Saved {yaml_file.resolve()}")


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


def write_to_json_pd(df: pd.DataFrame, json_file: Path) -> None:
  """
  Write query and response data to a JSON file.

  Args:
      df (pd.DataFrame): The DataFrame containing query and response columns.
      json_file (str): Path to the JSON file to write.
  """
  data: list[dict[str, Any]] = df.to_json(orient="records")  # type: ignore
  with open(json_file, "w") as file:
    file.write(data)  # type: ignore
  print(f"Saved {json_file.resolve()}")


def main() -> None:
  resources_path = Path.cwd().joinpath("resources")
  db_path = resources_path.joinpath("database.db")
  db_manager = DatabaseManager(database_filepath=db_path)

  db_manager.connectFunc()
  data_frame = db_manager.execute("SELECT name, address, city, state, stars, categories from restaurants")
  # output_file = resources_path.joinpath("best_places_by_city_queries.json")
  # generate_best_places_by_city(db_manager, output_file)
  # db_manager.closeFunc()

  checkpoint = resources_path.joinpath("checkpoint.yaml")
  output_file = resources_path.joinpath("query_learning.json")
  generate_category_queries_for_all_cities(data_frame, output_file, checkpoint)


if __name__ == "__main__":
  import sys

  sys.exit(main())
