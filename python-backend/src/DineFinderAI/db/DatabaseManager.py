"""
This file is responsible for the interaction with the underlying database.
It is used to provide the testdata for the ML model.
"""

import sqlite3
from pathlib import Path
import pandas as pd


class DatabaseManager:
  """
  A class to manage database operations and optionally interact with a JSON data file.

  Attributes:
      database_filepath (Path): The file path to the database file.
      json_filepath (Path | None): The file path to a JSON data file (optional).
  """
  def __init__(self, database_filepath: Path, json_filepath: Path | None = None) -> None:
    """
    Initializes the DatabaseManager with the paths to the database file and an optional JSON data file.

    Args:
        database_filepath (Path): The file path to the database file.
        json_filepath (Path | None, optional): The file path to a JSON data file. Defaults to None.
    """
    self.database_filepath = database_filepath
    self.json_filepath = json_filepath

  def connectFunc(self) -> None:
    """ Establishes a connection to the database. """
    self.con = sqlite3.connect(self.database_filepath)

  def closeFunc(self) -> None:
    """ Closes the database connection. """
    self.con.close()

  def execute(self, query: str) -> pd.DataFrame:
    """
    Executes an SQL query and returns the result as a pandas DataFrame.

    Args:
        query (str): The SQL query string to execute on the database.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the SQL query.
    """
    return pd.read_sql_query(query, self.con)

  def insertData(self) -> None:
    """
    Reads Restaurant data from a JSON file, filters the entries and stores the entries into the database.

    The following filters are applied:
        - Only businesses with the "Restaurants" category are kept.
        - Businesses with certain categories (e.g., "Beauty & Spas", "Health & Medical", etc.) are excluded.
        - Businesses with missing or empty "name" or "address" fields are excluded.
        - Businesses with "Wellness" in their name are excluded.
    """
    if not self.json_filepath:
      print("Please provide a json file with data before trying to insert data...")
      return
    
    db = pd.read_json(self.json_filepath, lines=True)
    db = db[["business_id", "name", "address", "city", "state", "postal_code", "stars", "review_count", "categories"]]

    db = db.dropna()
    db = db[db["categories"].str.contains("Restaurants") == True]
    db = db[db["categories"].str.contains("Beauty & Spas") == False]
    db = db[db["categories"].str.contains("Health & Medical") == False]
    db = db[db["categories"].str.contains("Doctors") == False]
    db = db[db["categories"].str.contains("Towing") == False]
    db = db[db["categories"].str.contains("Keys & Locksmith") == False]

    db = db[db["name"].str.contains("Wellness") == False]

    db = db[db["address"] != ""]

    db = self.removeCategory(db, ["Restaurants"])

    db.to_sql("restaurants", self.con)
    
  def removeCategory(self, df: pd.DataFrame, to_be_removed: list[str]) -> pd.DataFrame:
    """
    Remove keywords, specified in `to_be_removed`, from the categories string.

    Args:
        df (pd.DataFrame): Data from database file as pandas dataframe.
        to_be_removed (list[str]): A list with keywords which should be removed from every category string.

    Returns:
        pd.DataFrame: Adapted dataframe.
    """
    for index, row in df.iterrows():
      categories: str = row["categories"]
      category_list = categories.split(",")
      category_list = [category.strip() for category in category_list if category.strip() not in to_be_removed]
      df.at[index, 'categories'] = ", ".join(category_list)
      
    return df

  def addColumn(self, column_name: str, sql_data_type: str) -> None:
    """
    Adds a new column to the 'restaurants' table in the database.

    Args:
        column_name (str): The name of the column to add.
        sql_data_type (str): The SQL data type of the new column (e.g., 'TEXT', 'INTEGER').
    """
    query = f"ALTER TABLE restaurants ADD {column_name} {sql_data_type}"
    self.con.cursor().execute(query)

def confirmDeletion() -> bool:
  """
  Prompts the user to confirm whether they want to delete the existing database file.

  Returns:
      bool: Returns `True` if the user confirms deletion by entering 'y', and `False` if 
            the user declines by entering 'n'.
  """
  confirm: str = input("Do you want to delete the existing database file? [y|n] -> ")
  while confirm != "y" and confirm != "n":
    confirm = input("Please type either 'y' or 'n' -> ")

  return confirm == "y"

def main() -> None:
  resources_path = Path.cwd().joinpath("resources")
  db_path = resources_path.joinpath("database.db")
  data_path = resources_path.joinpath("yelp_academic_dataset_business.json")
  if not data_path.exists():
    print("Data path doesn't exist, please provide one.")
    return
  
  if db_path.exists() and not confirmDeletion():
    print("Abort")
    return

  db_path.unlink(missing_ok=True)

  db_manager = DatabaseManager(db_path, data_path)
  db_manager.connectFunc()
  db_manager.insertData()
  db_manager.closeFunc()

if __name__ == "__main__":
  import sys

  sys.exit(main())