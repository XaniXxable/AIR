"""
"""

import pathlib
from DatabaseManager import DatabaseManager
import sys

database_path = pathlib.Path("database.db")
database_path.unlink(missing_ok=True)

db_manager = DatabaseManager(database_path, pathlib.Path("yelp_academic_dataset_business.json"))
db_manager.connectFunc()
db_manager.insertData()
db_manager.closeFunc()