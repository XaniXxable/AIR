from FastDineAPI.recomentation_system.RestaurantNER import RestaurantNER
import pandas as pd
from pathlib import Path


def load_from_db() -> pd.DataFrame:
  import sqlite3

  db_loc = Path.cwd().joinpath("resources", "database.db")
  with sqlite3.connect(db_loc) as conn:
    query = "SELECT city, categories FROM restaurants"
    df = pd.read_sql_query(query, conn)
  return df


def main() -> None:
  data = pd.DataFrame()
  data = load_from_db()
  found_city_records = []
  ner_model_path = Path.cwd().joinpath("resources/model/NER/")
  ner_system = RestaurantNER(ner_model_path, data, custom_tokenizer=True)
  save_location = Path.cwd().joinpath("resources/found_categories.csv")
  data = data.drop_duplicates(subset=["city"])
  for _, row in data.drop_duplicates().iterrows():
    city = row["city"]
    query = f"Can you recommend any italian in {city}"
    entities: pd.DataFrame = ner_system._extract_entities(query)
    location = (
      " ".join([word for word, label in entities if label == "LOC"])
      .replace("[SEP]", "")
      .replace("[CLS]", "")
      .replace("?", "")
      .strip()
    )
    print(f"Checking for {city.lower()}")
    found = 1 if city.lower() == location else 0
    found_city_records.append({"city": city, "found": found})

  found_city = pd.DataFrame(found_city_records)
  found_city.to_csv(save_location, index=False)

  print(f"Data has been saved to {save_location}.")


if __name__ == "__main__":
  import sys

  sys.exit(main())
