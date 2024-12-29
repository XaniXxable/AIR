import yaml
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Callable
from DineFinderAI.db.DatabaseManager import DatabaseManager
import pandas as pd
import json


class QueryGenerator:
  def __init__(self) -> None:
    pass

  def _generate_by_name(self, names: list[str]) -> list[str]:
    queries = []
    for name in names:
      queries.append(f"What are the details for {name}?")
      queries.append(f"Show me reviews for {name}.")
      queries.append(f"Is there a {name} nearby?")
    return queries

  def _generate_by_category(self, categories: list[str], states: list[str], cities: list[str]) -> list[str]:
    queries = []
    for category in categories:
      for city in cities:
        queries.append(f"Where can I find good {category} in {city}?")
        queries.append(f"Best {category} places in {city}.")
      for state in states:
        queries.append(f"Best {category} places in {state}.")
    return queries

  def _generate_by_star(self, stars: list[float], categories: list[str]) -> list[str]:
    queries = []
    for star in stars:
      queries.append(f"List restaurants with at least {star} stars.")
      for category in categories:
        queries.append(f"Are there any {category} spots rated {star} stars or higher?")
    return queries

  def _generate_by_review(self, reviews: list[int], categories: list[str]) -> list[str]:
    queries = []
    for review in reviews:
      for category in categories:
        queries.append(f"Show top-rated {category} places with more than {review} reviews.")
    return queries

  def _generate_by_city_categ_stars(self, cities: list[str], categories: list[str], stars: list[str]) -> list[str]:
    queries = []
    for city in cities:
      for category in categories:
        for star in stars:
          queries.append(f"Find {category} restaurants in {city} with has more than {star} stars.")
    return queries

  def _generate_category_by_city(self, categories: list[str], cities: list[str]) -> list[str]:
    queries = []
    for category in categories:
      for city in cities:
        queries.append(f"I’m looking for the best {category} in {city}.")
        queries.append(f"What’s a highly rated {category} shop in {city}?")
        queries.append(f"Can you suggest a place for {category} with great reviews?")
    return queries

  def _generate_category_by_city_and_name(
    self, categories: list[str], cities: list[str], names: list[str]
  ) -> list[str]:
    queries = []
    for category in categories:
      for city in cities:
        queries.append(f"What’s the address of the best {category} spot in {city}?")
      for name in names:
        queries.append(f"Where is {name} located, and what are the reviews?")
    return queries

  def _generate_queries_from_function(self, func: Callable[..., list[str]], args: list) -> list[str]:
    return func(*args)

  def __call__(
    self,
    categories: list[str],
    cities: list[str],
    states: list[str],
    stars: list[int],
    reviews: list[int],
    names: list[str],
  ) -> list[str]:
    queries = []

    with Pool(processes=cpu_count()) as pool:
      results = pool.starmap(
        self._generate_queries_from_function,
        [
          (self._generate_by_name, [names]),
          (self._generate_by_category, [categories, states, cities]),
          (self._generate_by_star, [stars, categories]),
          (self._generate_by_review, [reviews, categories]),
          (self._generate_by_city_categ_stars, [cities, categories, stars]),
          (self._generate_category_by_city, [categories, cities]),
          (
            self._generate_category_by_city_and_name,
            [categories, cities, names],
          ),
        ],
      )

      for result in results:
        queries.extend(result)

    return queries


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


def process_category_queries_for_all_cities(
  df: pd.DataFrame, output_file: str, checkpoint_file: str, batch_size: int = 100
):
  """
  Process 'Where can I find good {category} in {city}?' queries with scores,
  saving progress in intervals to a YAML file and using a checkpoint file.

  Args:
      df (pd.DataFrame): DataFrame containing restaurant details.
      output_file (str): YAML file path to save the results.
      checkpoint_file (str): File path to save the checkpoint (last processed index).
      batch_size (int): Number of rows to process before saving progress.
  """
  # Normalize city and category columns to ensure case-insensitivity
  df["city_normalized"] = df["city"].str.lower()
  df["categories_normalized"] = df["categories"].str.lower().str.split(", ")

  # Create all unique city-category combinations
  city_category_combinations = pd.MultiIndex.from_product(
    [df["city_normalized"].unique(), df["categories_normalized"].explode().unique()], names=["city", "category"]
  ).to_frame(index=False)

  # Determine the starting index
  start_index = 0
  if Path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
      start_index = int(f.read().strip())

  print(f"Resuming from index: {start_index}")

  for i in range(start_index, len(city_category_combinations), batch_size):
    # Process a batch
    batch = city_category_combinations.iloc[i : i + batch_size]

    # Merge with the original DataFrame to check for matches
    batch_results = {}
    for _, row in batch.iterrows():
      city = row["city"]
      query_category = row["category"]
      responses = []
      scores = []

      if query_category in ["restaurants", "food", "nightlife"]:
        continue
      # Filter restaurants in the given city
      city_restaurants = df[df["city_normalized"] == city]
      cat_rest = filter_rows_by_string(city_restaurants, "categories", query_category)

      if cat_rest.empty:
        query = f"Where can I find good {query_category} in {city}?"
        response = f"Sorry, we couldn't find good {query_category} in {city}."
        batch_results[query] = {"responses": response, "scores": 0}
        continue

      query = f"Where can I find good {query_category} in {city}?"

      for _, restaurant in cat_rest.iterrows():
        response = f"You can find good {query_category} in {city} at {restaurant['name']}"
        score = count_matching_categories(restaurant["categories"], query_category)
        responses.append(response)
        scores.append(score)
      batch_results[query] = {"responses": responses, "scores": scores}

    write_to_json(batch_results, output_file)

    with open(checkpoint_file, "w") as checkpoint:
      checkpoint.write(str(i + batch_size))

    print(f"Processed up to index: {i + batch_size}")
    batch_results.clear()

  print("Processing completed.")


def count_matching_categories(target_categories: str, search_categories: str) -> int:
  matches = 0
  targets = target_categories.split(",")
  searches = search_categories.split(", ")
  for target_category in targets:
    for search_category in searches:
      if target_category.lower().strip() == search_category.lower().strip():
        matches += 1

  return matches / len(targets)


def write_to_yaml(df: pd.DataFrame, yaml_file: Path) -> None:
  """
  Write query and response data to a YAML file.

  Args:
      df (pd.DataFrame): The DataFrame containing query and response columns.
      yaml_file (str): Path to the YAML file to write.
  """
  data: list[dict[str, any]] = df[["query", "response"]].to_dict(orient="records")

  with open(yaml_file, "w") as file:
    yaml.dump(data, file, default_flow_style=False)

  print(f"Saved {yaml_file.resolve()}")


def write_to_json(data: dict[str, any], json_file: Path) -> None:
  """
  Write query and response data to a JSON file.

  Args:
      data (dict[str, any]): The dict containing query and response columns.
      json_file (str): Path to the JSON file to write.
  """
  with open(json_file, "a") as file:
    file.write(json.dumps(data))
  print(f"Saved {json_file.resolve()}")


def write_to_json_pd(df: pd.DataFrame, json_file: Path) -> None:
  """
  Write query and response data to a JSON file.

  Args:
      df (pd.DataFrame): The DataFrame containing query and response columns.
      json_file (str): Path to the JSON file to write.
  """
  data: list[dict[str, any]] = df.to_json(orient="records")
  with open(json_file, "w") as file:
    file.write(data)
  print(f"Saved {json_file.resolve()}")


def main() -> None:
  resources_path = Path.cwd().joinpath("resources")
  db_path = resources_path.joinpath("database.db")
  db_manager = DatabaseManager(database_filepath=db_path)
  db_manager.connectFunc()
  data_frame = db_manager.execute("SELECT name, address, city, state, stars, categories from restaurants")
  db_manager.closeFunc()

  name_file_path = resources_path.joinpath("name_detail_queries_pandas.yaml")
  checkpoint = resources_path.joinpath("checkpoint.yaml")
  output_file = resources_path.joinpath("categ_in_city_queries.json")
  rest_details = process_category_queries_for_all_cities(data_frame, output_file, checkpoint)

  print(f"Generated {len(data_frame)} query-response pairs")
  write_to_json(rest_details, name_file_path)

  # categories = ["bubble tea", "ice cream", "Italian", "Mexican", "Chinese"]
  # cities = ["New York", "Los Angeles", "Chicago"]
  # states = ["NY", "CA", "IL"]
  # star_thresholds = [3, 4, 5]
  # review_thresholds = [50, 100, 200]
  # names = ["Joe's Pizza", "Shake Shack", "The Great Wall"]

  # gen = QueryGenerator()
  # queries = gen(categories, cities, states, star_thresholds, review_thresholds, names)

  # dest_file_path = Path.cwd().joinpath("resources", "queries.yaml")
  # with open(dest_file_path, "w") as f:
  #   yaml.dump({"queries": queries}, f, default_flow_style=False)

  # print(f"Generated {len(queries)} queries and saved to '{dest_file_path}'.")


if __name__ == "__main__":
  import sys

  sys.exit(main())
