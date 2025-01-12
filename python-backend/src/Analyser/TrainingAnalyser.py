from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


class TrainingAnalyser:
    def __init__(self, logging_dir: Path) -> None:
        self.data = pd.read_json(logging_dir)

    def plot(self) -> None:
        plt.figure(figsize=(10, 6))

        plt.plot(self.data["epoch"], self.data["eval_loss"], label="Eval Loss")

        plt.title("Evaluation Loss Over Folds")
        plt.xlabel("Fold")
        plt.ylabel("Evaluation Loss")
        plt.legend()
        plt.show()


def main() -> None:
    """
    Entry point of the script.
    """
    logging_path = Path.cwd().joinpath("resources/model/training/")
    runtime_analyser = TrainingAnalyser(logging_path)

    runtime_analyser.plot()


if __name__ == "__main__":
    import sys

    sys.exit(main())
