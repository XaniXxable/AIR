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
    db = pd.read_json(self.json_filepath, lines=True)
    db = db[["name", "address", "city", "state", "postal_code", "stars", "review_count", "categories"]]
    db = db[db["categories"].str.contains("Restaurants") == True]
    db.to_sql("restaurants", self.con)
