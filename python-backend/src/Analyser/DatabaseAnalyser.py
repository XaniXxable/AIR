from pathlib import Path
from DineFinderAI.db.DatabaseManager import DatabaseManager
import matplotlib.pyplot as plt
import numpy as np

class DatabaseAnalyser:
  def __init__(self, db_manager: DatabaseManager, findings_folder_path: Path) -> None:
    self.db_manager = db_manager
    self.findings_folder_path = findings_folder_path

  def __call__(self) -> None:
    self.findings_folder_path.mkdir(exist_ok=True)
    self.db_manager.connectFunc()
    self._plot_cities_with_most_restaurants()
    self._plot_important_categories(["Italian", "Mexican", "Chinese", "Traditional", "American", "Korean", "Greek", "Japanese", "Fast Food"])
    self.db_manager.closeFunc()
  
  def _plot_cities_with_most_restaurants(self):
    query = "SELECT city FROM restaurants"
    df = self.db_manager.execute(query)
    restaurant_by_city_counter: dict[str, int] = {}

    for _, row in df.iterrows():
      city: str = row["city"]
      city = city.capitalize()
      if city not in restaurant_by_city_counter.keys():
        restaurant_by_city_counter[city] = 1
        continue
      restaurant_by_city_counter[city] = restaurant_by_city_counter[city] + 1

    restaurant_by_city_counter = dict(sorted(restaurant_by_city_counter.items(), key=lambda item: item[1], reverse=True))

    num_cities = 10
    cities = list(restaurant_by_city_counter.keys())[0:num_cities]
    counters = list(restaurant_by_city_counter.values())[0:num_cities]
    # Get colors from the 'tab20' colormap
    colors = plt.cm.tab20(np.linspace(0, 1, len(cities)))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,8))
    ax.bar(cities, counters, color=colors)
    ax.set_title(f'Top {num_cities} Cities with the most restaurants')
    ax.set_xlabel('Cities')
    ax.set_ylabel('Counts')
    output_path = self.findings_folder_path.joinpath("cities_with_most_restaurants.pdf")
    fig.savefig(fname=str(output_path), format="pdf")
  
  def _plot_important_categories(self, categories: list[str]):
    query = "SELECT categories FROM restaurants"
    df = self.db_manager.execute(query)
    categories_counter: dict[str, int] = {}

    for _, row in df.iterrows():
      categories_text: str = row["categories"]
      for category in categories:
        if category in categories_text:
          if category not in categories_counter.keys():
            categories_counter[category] = 1
            continue
          categories_counter[category] = categories_counter[category] + 1
          categories_counter[category]

    categories_counter = dict(sorted(categories_counter.items(), key=lambda item: item[1], reverse=True))

    category_names = list(categories_counter.keys())
    count = list(categories_counter.values())
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,10))
    ax.scatter(category_names, count, s=[c * 0.2 for c in count], c=np.arange(len(category_names)), cmap='tab20', alpha=0.5)
    ax.set_title('tba')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Count')
    output_path = self.findings_folder_path.joinpath("important_categories.pdf")
    fig.savefig(fname=str(output_path), format="pdf")
  

def main() -> None:
  resources_path = Path.cwd().joinpath("resources")
  findings_folder_path = resources_path.joinpath("findings")
  db_path = resources_path.joinpath("database.db")
  db_analyser = DatabaseAnalyser(DatabaseManager(db_path), findings_folder_path)

  db_analyser()

if __name__ == "__main__":
  import sys

  sys.exit(main())