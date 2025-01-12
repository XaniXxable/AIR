from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class RuntimeAnalyser:
  """
  A class for analyzing runtime data and storing findings.
  """

  def __init__(self, findings_folder_path: Path) -> None:
    """
    Initializes the RuntimeAnalyser with a folder path for the findings.

    Args:
        findings_folder_path (Path): Path to store runtime analysis results.
    """
    self.findings_folder_path = findings_folder_path

  def __call__(self) -> None:
    """
    Executes the runtime analysis workflow.
    """
    found_cities_path = self.findings_folder_path.joinpath("found_cities.csv")
    found_categories_path = self.findings_folder_path.joinpath("found_categories.csv")

    if not found_cities_path.is_file() or not found_categories_path.is_file():
      print("No files found for runtime analysis. Please run 'poetry run analyse_queries' to get the required files...")
      return

    self._analyse_runtime_of_cities_queries(found_cities_path)
    self._analyse_runtime_of_categories_queries(found_categories_path)

  def _analyse_runtime_of_cities_queries(self, found_cities_path: Path):
    """
    Analyzes the runtime of city queries and generates a runtime plot.

    Args:
        found_cities_path (Path): Path to the CSV file containing city query runtimes.
    """
    df = pd.read_csv(found_cities_path)
    runtimes: list[float] = [runtime * 1e-6 for runtime in df["delta [ns]"].to_list()]
    mean_runtime = np.mean(runtimes)
    std_runtime = np.std(runtimes)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 7))
    ax.plot(runtimes)
    ax.set_title(f"Runtimes for processing a city query (μ={mean_runtime:.2f} ms, σ={std_runtime:.2f} ms)", fontsize=18)
    ax.set_xlabel("Query number", fontsize=14)
    ax.set_ylabel("Duration [ms]", fontsize=14)
    ax.tick_params(axis="x", which="major", labelsize=11)
    ax.tick_params(axis="y", which="major", labelsize=11)
    ax.grid(True)

    output_path = self.findings_folder_path.joinpath("city_queries_runtime.png")
    fig.savefig(fname=str(output_path), format="png")

    print(f"Saved {output_path.resolve()}")

  def _analyse_runtime_of_categories_queries(self, found_categories_path: Path):
    """
    Analyzes the runtime of category queries and generates a runtime plot.

    Args:
        found_categories_path (Path): Path to the CSV file containing category query runtimes.
    """
    df = pd.read_csv(found_categories_path)
    runtimes: list[float] = [runtime * 1e-6 for runtime in df["delta [ns]"].to_list()]
    mean_runtime = np.mean(runtimes)
    std_runtime = np.std(runtimes)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 7))
    ax.plot(runtimes)
    ax.set_title(
      f"Runtimes for processing a category query (μ={mean_runtime:.2f} ms, σ={std_runtime:.2f} ms)", fontsize=18
    )
    ax.set_xlabel("Query number", fontsize=14)
    ax.set_ylabel("Duration [ms]", fontsize=14)
    ax.tick_params(axis="x", which="major", labelsize=11)
    ax.tick_params(axis="y", which="major", labelsize=11)
    ax.grid(True)

    output_path = self.findings_folder_path.joinpath("category_queries_runtime.png")
    fig.savefig(fname=str(output_path), format="png")

    print(f"Saved {output_path.resolve()}")


def main() -> None:
  """
  Entry point of the script.
  """
  resources_path = Path.cwd().joinpath("resources")
  findings_folder_path = resources_path.joinpath("findings")
  runtime_analyser = RuntimeAnalyser(findings_folder_path)

  runtime_analyser()


if __name__ == "__main__":
  import sys

  sys.exit(main())
