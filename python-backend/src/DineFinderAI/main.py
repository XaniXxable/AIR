from interface.response.queryResponse import QueryResponse
from interface.request.queryRequest import QueryRequest
from DineFinderAI.models.DineFinder import DineFinder
from DineFinderAI.models.FineTuning import FineTuning
from pathlib import Path
import json
import sys
from DineFinderAI.models.TokenAnalyser import TokenAnalyser


def load_json(file_name: str) -> dict[str, any]:
  resource_folder = Path.cwd().joinpath("resources")
  with open(resource_folder.joinpath(file_name), "r") as file:
    content = json.load(file)
  print(f"File {file_name} loaded")
  return content


def main() -> None:
  data = load_json("ner_training_dataset.json")
  analyser = TokenAnalyser(new_tokens=True)
  analyser.train(data)


def combine_json() -> None:
  resource = Path.cwd().joinpath("resources")
  best_places = resource.joinpath("best_places_by_city_queries.json")
  categ_in_city = resource.joinpath("categ_in_city_queries.json")

  with open(best_places, "r") as file1:
    data1 = json.load(file1)
  with open(categ_in_city, "r") as file2:
    data2 = json.load(file2)

  combined_data = {**data1, **data2}
  output_path = resource.joinpath("trainingset.json")

  with open(output_path, "w") as output_file:
    json.dump(combined_data, output_file, indent=4)

  print(f"Combined JSON saved to {output_path}")


if __name__ == "__main__":
  sys.exit(main())
