from FastDineAPI.recomentation_system.RestaurantNER import RestaurantNER
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json
from time import perf_counter_ns

finding_folder = Path.cwd().joinpath("resources/findings")
finding_folder.mkdir(exist_ok=True)


def load_from_db() -> pd.DataFrame:
  import sqlite3

  db_loc = Path.cwd().joinpath("resources/database.db")
  with sqlite3.connect(db_loc) as conn:
    query = "SELECT city, categories FROM restaurants"
    df = pd.read_sql_query(query, conn)
  return df


def extract_not_found_cities() -> None:
  file_path = finding_folder.joinpath("found_categories.csv")
  new_tokens = Path.cwd().joinpath("resources/new_tokens.json")

  try:
    found_city = pd.read_csv(file_path)
  except FileNotFoundError:
    print(f"File not found at {file_path}. Please check the file path and try again.")
    return

  with open(new_tokens, "r") as file:
    tokens = json.load(file)

  not_found_cities = found_city.loc[found_city["found"] == 0, "city"].tolist()

  for city in not_found_cities:
    tokens["city"].append(city)

  with open(new_tokens, "w") as json_file:
    json.dump(tokens, json_file, indent=4)


def plot_restaurants_found() -> None:
  file_path = finding_folder.joinpath("found_cities.csv")
  try:
    found_city = pd.read_csv(file_path)
  except FileNotFoundError:
    print(f"File not found at {file_path}. Please check the file path and try again.")
    return

  # Group by 'found' to count how many cities were found vs not found
  found_summary = found_city["found"].value_counts()

  # Plot the results
  plt.figure(figsize=(8, 6))
  found_summary.plot(kind="bar", color=["skyblue", "salmon"])
  plt.title("Cities Found vs Not Found")
  plt.xlabel("Found Status (1 = Found, 0 = Not Found)")
  plt.ylabel("Number of Cities")
  plt.xticks(rotation=0)
  plt.tight_layout()

  plt.savefig(finding_folder.joinpath("found_categories.png"))


def plot_categories_found() -> None:
  file_path = finding_folder.joinpath("found_categories.csv")
  try:
    found_city = pd.read_csv(file_path)
  except FileNotFoundError:
    print(f"File not found at {file_path}. Please check the file path and try again.")
    return

  # Group by 'found' to count how many cities were found vs not found
  found_summary = found_city["found"].value_counts()

  # Plot the results
  plt.figure(figsize=(8, 6))
  found_summary.plot(kind="bar", color=["skyblue", "salmon"])
  plt.title("Categories Found vs Not Found")
  plt.xlabel("Found Status (1 = Found, 0 = Not Found)")
  plt.ylabel("Number of Categories")
  plt.xticks(rotation=0)
  plt.tight_layout()

  plt.savefig(finding_folder.joinpath("found_categories.png"))


def check_city_finding() -> None:
  plot_restaurants_found()
  data = pd.DataFrame()
  data = load_from_db()
  found_city_records = []
  ner_model_path = Path.cwd().joinpath("resources/model/NER/")
  ner_system = RestaurantNER(ner_model_path, data, custom_tokenizer=True)
  save_location = finding_folder.joinpath("found_cities.csv")
  data = data.drop_duplicates(subset=["city"])
  for _, row in data.drop_duplicates().iterrows():
    city = row["city"]
    query = f"Can you recommend any italian in {city}"
    start = perf_counter_ns()
    entities: pd.DataFrame = ner_system._extract_entities(query)
    end = perf_counter_ns()
    location = (
      " ".join([word for word, label in entities if label == "LOC"])
      .replace("[SEP]", "")
      .replace("[CLS]", "")
      .replace("?", "")
      .strip()
    )
    print(f"Checking for {city.lower()}")
    found = 1 if city.lower().strip() == location.lower().strip() else 0
    found_city_records.append({"city": city, "found": found, "delta [ns]": end - start})

  found_city = pd.DataFrame(found_city_records)
  found_city.to_csv(save_location, index=False)

  print(f"Data has been saved to {save_location}.")


def check_category_finding() -> None:
  plot_restaurants_found()
  data = pd.DataFrame()
  data = [
    "vegan",
    "plant-based",
    "dairy-free",
    "egg-free",
    "cruelty-free",
    "vegetarian",
    "meat-free",
    "lacto-ovo",
    "veggie options",
    "gluten-free",
    "celiac-friendly",
    "halal-certified",
    "halal meat",
    "kosher",
    "kosher-certified",
    "keto-friendly",
    "low carb",
    "paleo",
    "nut-free",
    "allergy-safe",
    "soy-free",
    "organic",
    "locally-sourced",
    "non-GMO",
    "low-fat",
    "healthy options",
    "sugar-free",
    "raw food",
    "uncooked",
    "raw vegan",
    "Italian",
    "Mexican",
    "Indian",
    "Japanese",
    "Chinese",
    "French",
    "fusion",
    "modern twist",
    "innovative cuisine",
    "street food",
    "food trucks",
    "hawker-style",
    "dessert parlor",
    "bakery",
    "patisserie",
    "coffee shop",
    "wine bar",
    "craft beer",
    "tea house",
    "fast food",
    "quick bites",
    "drive-thru",
    "casual dining",
    "diner",
    "mid-range",
    "pop-up restaurant",
    "seasonal",
    "temporary",
    "chain restaurant",
    "franchise",
  ]
  found_category_records = []
  ner_model_path = Path.cwd().joinpath("resources/model/NER/")
  ner_system = RestaurantNER(ner_model_path, data, custom_tokenizer=True)
  save_location = finding_folder.joinpath("found_categories.csv")

  for category in data:
    query = f"Can you recommend any {category} in Graz"
    start = perf_counter_ns()
    entities: pd.DataFrame = ner_system._extract_entities(query)
    end = perf_counter_ns()
    tmp = (
      ", ".join([word for word, label in entities if label == "CUISINE" and word != ","])
      .replace("[SEP]", "")
      .replace("[CLS]", "")
      .strip()
    )
    print(f"Checking for {category.lower()}")
    found = 1 if tmp.lower().strip() in category.lower().strip() else 0
    found_category_records.append({"category": category, "found": found, "delta [ns]": end - start})

    found_categories = pd.DataFrame(found_category_records)
    found_categories.to_csv(save_location, index=False)

  print(f"Data has been saved to {save_location}.")


def main() -> None:
  # check_city_finding()
  # plot_restaurants_found()
  # extract_not_found_cities()
  # check_category_finding()
  plot_categories_found()


if __name__ == "__main__":
  import sys

  sys.exit(main())
