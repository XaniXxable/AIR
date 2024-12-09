"""
  This file is responsible for the interaction with the underlying database.
  It is used to provide the testdata for the ML model.
"""

import sqlite3
import pathlib
import pandas

class DatabaseManager:
  
  def __init__(self, database_filepath: pathlib.Path, json_filepath: pathlib.Path) -> None:
    self.database_filepath = database_filepath
    self.json_filepath = json_filepath

  def connectFunc(self) -> None:
    self.con = sqlite3.connect(self.database_filepath)

  def closeFunc(self) -> None:
    self.con.close()

  def initTable(self) -> None:
    self.con.cursor().execute("CREATE TABLE restaurants (id integer PRIMARY KEY AUTOINCREMENT, name varchar(255), address varchar(255), city varchar(255), state varchar(255), postal_code varchar(255), stars float, review_count int, categories varchar(255))")

  def insertData(self) -> None:
    db = pandas.read_json(self.json_filepath, lines=True)
    db = db[["name", "address", "city", "state", "postal_code", "stars", "review_count", "categories"]]
    db = db[db['categories'].str.contains("Restaurants")==True]
    db.to_sql("restaurants", self.con)