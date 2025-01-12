from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


class TrainingAnalyser:
  """
  A class for analyzing eval loss data and storing findings.
  """

  def __init__(self, logging_dir: Path) -> None:
    self.data = pd.read_json(logging_dir)
    self.data.dropna(subset=["eval_loss"], inplace=True)

  def plot(self, output_path: Path) -> None:
    """
    Analyzes the eval loss of training and generates a plot.

    Args:
        output_path (Path): Path to the JSON file containing the training metrics.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 7))
    ax.plot(self.data["epoch"], self.data["eval_loss"], label="Eval Loss")
    ax.set_title("Evaluation Loss Over Epochs", fontsize=18)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Evaluation Loss", fontsize=14)
    ax.tick_params(axis="x", which="major", labelsize=11)
    ax.tick_params(axis="y", which="major", labelsize=11)
    ax.legend()

    file_format = output_path.name.split(".")[-1]
    fig.savefig(fname=str(output_path), format=file_format)
    print(f"Saved {output_path.resolve()}")


def main() -> None:
  """
  Entry point of the script.
  """
  logging_path = Path.cwd().joinpath("resources", "model", "training", "fold_0", "training_logs.json")
  output_path = Path.cwd().joinpath("resources", "findings", "eval_loss.png")
  runtime_analyser = TrainingAnalyser(logging_path)

  runtime_analyser.plot(output_path)


if __name__ == "__main__":
  import sys

  sys.exit(main())
