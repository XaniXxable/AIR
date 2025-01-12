from DineFinderAI.models.SentimentModelTrainer import SentimentModelTrainer
from pathlib import Path
import pandas as pd
import random
import matplotlib.pyplot as plt


class SentimentAnalyser:
  LABELS = {"Positive": 0, "Neutral": 1, "Negative": 2}

  def __init__(self, generate: bool = False) -> None:
    self.reviews = Path.cwd().joinpath("resources/restaurant_reviews_checker.csv")
    model_path = Path.cwd().joinpath("resources/model/SEM_model")
    self.num_labels = len(self.LABELS.items())
    self.model = SentimentModelTrainer(model_name=model_path)
    if generate:
      self.data: pd.DataFrame = self.generate_restaurant_reviews()
      self.data.to_csv(self.reviews)
      return
    self.data = pd.read_csv(self.reviews)

  def generate_restaurant_reviews(self, num_reviews=500):
    positive_phrases = [
      "The food was outstanding!",
      "Such a delightful experience, I will definitely come back.",
      "Amazing service and a cozy ambiance.",
      "Best meal I've had in a long time.",
      "Perfect spot for a family dinner or special occasion.",
      "The dessert was absolutely heavenly.",
      "Great variety on the menu and everything was delicious.",
      "Highly recommend this place to everyone!",
      "Exceptional dining experience, from start to finish.",
      "The waiter was kind, and the food was served hot and fresh.",
    ]

    neutral_phrases = [
      "The experience was average, nothing remarkable.",
      "The food was okay, but I’ve had better.",
      "An ordinary restaurant with decent service.",
      "The ambiance was nice, but the food was just fine.",
      "Nothing special, but it wasn’t bad either.",
      "A very standard dining experience overall.",
      "It was fine, but I wouldn’t rush back.",
      "The service was okay, but it took a while to get seated.",
      "The food portions were fair, but the taste was mediocre.",
      "An average restaurant experience with no surprises.",
    ]

    negative_phrases = [
      "The food was cold and lacked flavor.",
      "Terrible experience, I wouldn’t recommend this place.",
      "The staff were rude, and the service was slow.",
      "Extremely overpriced for the quality of food.",
      "The restaurant was too noisy, and the chairs were uncomfortable.",
      "The portions were tiny, and the food was overcooked.",
      "I waited for over an hour, and the food was disappointing.",
      "The dessert was stale, and the coffee was burnt.",
      "A very disappointing meal, will not return.",
      "Worst dining experience I've had in years.",
    ]

    reviews = []
    sentiments = []
    for _ in range(num_reviews):
      sentiment = random.choice(["Positive", "Neutral", "Negative"])

      match sentiment:
        case "Positive":
          review = random.choice(positive_phrases)
        case "Neutral":
          review = random.choice(neutral_phrases)
        case "Negative":
          review = random.choice(negative_phrases)
        case _:
          raise NotImplementedError(f"Sentiment '{sentiment}' not implemented!")

      reviews.append(review)
      sentiments.append(sentiment)

    return pd.DataFrame({"Review": reviews, "Sentiment": sentiments})

  def analyse(self) -> None:
    save_location = Path.cwd().joinpath("resources/findings/reviews_analyse.csv")
    findings = []
    labels = {0: "Positive", 1: "Neutral", 2: "Negative"}
    predict = self.model.predict(self.data["Review"].tolist())
    actual_labels = self.data["Sentiment"].tolist()
    for index, review in enumerate(self.data["Review"].tolist()):
      tmp = predict[index]
      score = tmp["probabilities"]
      predict_label = labels[tmp["class"]]

      findings.append(
        {
          "review": review,
          "score": max(score.values()),
          "predict_label": predict_label,
          "actual_label": actual_labels[index],
        }
      )

    data = pd.DataFrame(findings)
    data.to_csv(save_location, index=False)

    print(f"Data has been saved to {save_location}.")


def main() -> None:
  # analyser = SentimentAnalyser(generate=True)
  # analyser.analyse()

  save_location = Path.cwd().joinpath("resources/findings/reviews_analyse.csv")
  data = pd.read_csv(save_location)

  predicted_counts = data["predict_label"].value_counts()
  actual_counts = data["actual_label"].value_counts()

  categories = ["Positive", "Neutral", "Negative"]
  predicted_counts = predicted_counts.reindex(categories, fill_value=0)
  actual_counts = actual_counts.reindex(categories, fill_value=0)

  # Plot the counts
  plt.figure(figsize=(8, 6))
  plt.bar(categories, predicted_counts, alpha=0.7, label="Predicted")
  plt.bar(categories, actual_counts, alpha=0.7, label="Actual", edgecolor="black")
  plt.title("Comparison of Predicted and Actual Labels")
  plt.xlabel("Sentiment Labels")
  plt.ylabel("Count")
  plt.legend()
  plt.grid(axis="y", linestyle="--", alpha=0.7)
  plt.savefig(Path.cwd().joinpath("resources/findings/sentiment_plot.png"))


if __name__ == "__main__":
  import sys

  sys.exit(main())
