from pathlib import Path
import json
import sys
from DineFinderAI.models.TokenAnalyser import TokenAnalyser
from typing import Any


def load_json(file_name: str) -> dict[str, Any]:
  """
  Loads JSON data from a specified file in the 'resources' directory.

  Args:
      file_name (str): Name of the JSON file to load.

  Returns:
      dict[str, Any]: Parsed content of the JSON file.
  """
  resource_folder = Path.cwd().joinpath("resources")
  with open(resource_folder.joinpath(file_name), "r") as file:
    content = json.load(file)
  print(f"File {file_name} loaded...")
  return content


def main() -> None:
  """
  Entry point of the script.
  """
  data = load_json("ner_training_dataset.json")
  analyser = TokenAnalyser(new_tokens=True)
  analyser.train(data)

if __name__ == "__main__":
  sys.exit(main())
