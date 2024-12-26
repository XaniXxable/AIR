"""
This file is responsible for the interaction with the underlying database.
It is used to provide the testdata for the ML model.
"""

import sqlite3
import pathlib
import pandas as pd
from typing import Tuple, Any


class DatabaseManager:
  def __init__(self, database_filepath: pathlib.Path, json_filepath: pathlib.Path | None = None) -> None:
    self.database_filepath = database_filepath
    self.json_filepath = json_filepath

  def connectFunc(self) -> None:
    self.con = sqlite3.connect(self.database_filepath)

  def closeFunc(self) -> None:
    self.con.close()

  def execute(self, query: str) -> pd.DataFrame:
    return pd.read_sql_query(query, self.con)

  def insertData(self) -> None:
    if not self.json_filepath:
      print("Please provide a json file with data before trying to insert data...")
      return
    
    db = pd.read_json(self.json_filepath, lines=True)
    db = db[["name", "address", "city", "state", "postal_code", "stars", "review_count", "categories"]]

    db = db.dropna()
    db = db[db["categories"].str.contains("Restaurants") == True]
    db = db[db["categories"].str.contains("Beauty & Spas") == False]
    db = db[db["categories"].str.contains("Health & Medical") == False]
    db = db[db["categories"].str.contains("Doctors") == False]
    db = db[db["categories"].str.contains("Towing") == False]
    db = db[db["categories"].str.contains("Keys & Locksmith") == False]

    db = db[db["name"].str.contains("Wellness") == False]
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

