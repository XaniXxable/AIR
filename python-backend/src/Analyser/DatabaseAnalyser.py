from pathlib import Path
from DineFinderAI.db.DatabaseManager import DatabaseManager
import matplotlib.pyplot as plt
import numpy as np

class DatabaseAnalyser:
  """
  A class for analyzing database content and storing findings.
  """
  def __init__(self, db_manager: DatabaseManager, findings_folder_path: Path) -> None:
    """
    Initializes the DatabaseAnalyser with a DatabaseManager and folder path for the findings.

    Args:
        db_manager (DatabaseManager): The DatabaseManager instance for database operations.
        findings_folder_path (Path): Path to store analysis results.
    """
    self.db_manager = db_manager
    self.findings_folder_path = findings_folder_path

  def __call__(self) -> None:
    """
    Executes the database analysis workflow.
    """
    self.findings_folder_path.mkdir(exist_ok=True)
    self.db_manager.connectFunc()
    self._plot_cities_with_most_restaurants()
    imoortant_categories = ["Italian", "Mexican", "Chinese", "Traditional", "American", "Korean", "Greek", "Japanese", "Fast Food", "Burgers"]
    self._plot_important_categories(imoortant_categories)
    self.db_manager.closeFunc()

  def _save_bar_chart(self, x_values: list[str], y_values: list[int], title: str, xlabel: str, ylabel: str, output_path: Path):
    """
    Saves a bar chart.

    Args:
        x_values (list[str]): Labels for the x-axis.
        y_values (list[int]): Values for the y-axis.
        title (str): Title of the bar chart.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        output_path (Path): File path to save the chart.
    """
    # Get colors from the 'tab20' colormap
    colors = plt.cm.tab20(np.linspace(0, 1, len(y_values)))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,7))
    ax.bar(x_values, y_values, color=colors)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='x', which='major', labelsize=11)
    ax.tick_params(axis='y', which='major', labelsize=11)

    file_format = output_path.name.split(".")[-1]
    fig.savefig(fname=str(output_path), format=file_format)
    print(f"Saved {output_path.resolve()}")

  def _save_bubble_chart(self, x_values: list[str], y_values: list[int], title: str, xlabel: str, ylabel: str, output_path: Path):
    """
    Saves a bubble chart.

    Args:
        x_values (list[str]): Labels for the x-axis.
        y_values (list[int]): Values for the y-axis (used for both position and bubble size).
        title (str): Title of the bubble chart.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        output_path (Path): File path to save the chart.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,7))
    ax.scatter(x_values, y_values, s=[c * 0.2 for c in y_values], c=np.arange(len(x_values)), cmap='tab20', alpha=0.5)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_ylim(top=max(y_values) + max(y_values) * 0.1)
    ax.tick_params(axis='x', which='major', labelsize=11)
    ax.tick_params(axis='y', which='major', labelsize=11)

    file_format = output_path.name.split(".")[-1]
    fig.savefig(fname=str(output_path), format=file_format)
    print(f"Saved {output_path.resolve()}")
  
  def _plot_cities_with_most_restaurants(self):
    """
    Plots and saves a bar chart of the top 10 cities with the most restaurants.
    """
    query = "SELECT city FROM restaurants"
    df = self.db_manager.execute(query)
    restaurant_by_city_counter: dict[str, int] = {}

    for _, row in df.iterrows():
      city: str = row["city"]
      city = city.capitalize()
      self._increase_counter(restaurant_by_city_counter, city)

    restaurant_by_city_counter = dict(sorted(restaurant_by_city_counter.items(), key=lambda item: item[1], reverse=True))

    num_cities = 10
    cities = list(restaurant_by_city_counter.keys())[0:num_cities]
    counters = list(restaurant_by_city_counter.values())[0:num_cities]

    self._save_bar_chart(x_values=cities, 
                         y_values=counters,
                         title=f'Top {num_cities} Cities with the most restaurants in the dataset',
                         xlabel='Cities',
                         ylabel='Occurrences',
                         output_path=self.findings_folder_path.joinpath("cities_with_most_restaurants.png"))
  
  def _plot_important_categories(self, categories: list[str]):
    """
    Plots and saves a scatter plot in form of a bubble chart and a bar chart of specified restaurant categories by 
    their occurrence.

    Args:
        categories (list[str]): A list of important categories to analyze and plot.
    """
    query = "SELECT categories FROM restaurants"
    df = self.db_manager.execute(query)
    categories_counter: dict[str, int] = {}

    for _, row in df.iterrows():
      categories_text: str = row["categories"]
      for category in categories:
        if category in categories_text:
          self._increase_counter(categories_counter, category)

    categories_counter = dict(sorted(categories_counter.items(), key=lambda item: item[1], reverse=True))

    category_names = list(categories_counter.keys())
    counters = list(categories_counter.values())

    self._save_bar_chart(x_values=category_names, 
                         y_values=counters,
                         title='Occurrences of important categories in the dataset',
                         xlabel='Categories',
                         ylabel='Occurrences',
                         output_path=self.findings_folder_path.joinpath("important_categories_bar.png"))

    self._save_bubble_chart(x_values=category_names, 
                            y_values=counters,
                            title='Occurrences of important categories in the dataset',
                            xlabel='Categories',
                            ylabel='Occurrences',
                            output_path=self.findings_folder_path.joinpath("important_categories_bubble.png"))
  
  def _increase_counter(self, dictionary: dict[str, int], key: str) -> None:
    """
    Increments the count of a key in the given dictionary. If not present, initialise the counter for the given key 
    with 1.

    Args:
        dictionary (dict[str, int]): The dictionary to update.
        key (str): The key whose count needs to be incremented.
    """
    if key not in dictionary.keys():
      dictionary[key] = 1
      return
    dictionary[key] = dictionary[key] + 1

def main() -> None:
  """
  Entry point of the script.
  """
  resources_path = Path.cwd().joinpath("resources")
  findings_folder_path = resources_path.joinpath("findings")
  db_path = resources_path.joinpath("database.db")
  db_analyser = DatabaseAnalyser(DatabaseManager(db_path), findings_folder_path)

  db_analyser()

if __name__ == "__main__":
  import sys

  sys.exit(main())