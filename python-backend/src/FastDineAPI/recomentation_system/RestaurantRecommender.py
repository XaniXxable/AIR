from FastDineAPI.recomentation_system.RestaurantNER import RestaurantNER
import pandas as pd
from pathlib import Path


class RestaurantRecommenter:
  def __init__(self) -> None:
    self._ner_model_path = Path.cwd().joinpath("resources/model/NER/")
    self.data = self._load_from_db()
    self.ner_model = RestaurantNER(self._ner_model_path, self.data, custom_tokenizer=True)

  def __call__(self, query: str, feature_weight: dict[str, str] | None = None) -> pd.DataFrame:
    filtered_restaurants = self.ner_model(query)
    if filtered_restaurants.empty:
      return pd.DataFrame()
    final_filtered_df = self._calculate_scores(filtered_restaurants, feature_weights=feature_weight)
    top_10_restaurants = final_filtered_df.sort_values(by="score", ascending=False).head(10)
    return top_10_restaurants

  def _calculate_scores(self, filtered_df: pd.DataFrame, feature_weights: dict[str, int] | None = None) -> pd.DataFrame:
    """
    Calculate scores for the filtered DataFrame based on cuisine frequency.

    Args:
        filtered_df (pd.DataFrame): DataFrame containing filtered restaurants.
        feature_weights (dict[str, int] | None): Dictionary of feature weights with keys as feature names
                                                (e.g., 'pet_friendly', 'price_value_ratio') and values as
                                                dictionaries mapping sentiment ('positive', 'neutral', 'negative') to weights.

                                                Example:
                                                {
                                                    'pet_friendly': {'positive': 2, 'neutral': 0, 'negative': -1},
                                                    'price_value_ratio': {'positive': 3, 'neutral': 1, 'negative': -2}
                                                }
    Returns:
        pd.DataFrame: DataFrame with an additional 'score' column.
    """
    if filtered_df.empty:
      return filtered_df

    # Calculate frequency of each cuisine
    cuisine_counts = filtered_df["categories"].value_counts().to_dict()
    # Assign a score based on cuisine frequency
    filtered_df["score"] = filtered_df["categories"].apply(lambda x: cuisine_counts.get(x, 0))

    # Add scores based on labeled features
    highest_review_count = filtered_df.sort_values(by="review_count", ascending=False).iloc[0]["review_count"]
    filtered_df["normalized_rating"] = filtered_df["stars"] / 5.0
    filtered_df["normalized_review_count"] = filtered_df["review_count"] / highest_review_count

    filtered_df["score"] = (
      0.5 * filtered_df["normalized_rating"] + 0.5 * filtered_df["normalized_review_count"]
    )

    if feature_weights is None:
      return filtered_df

    for feature, sentiments in feature_weights.items():
      if feature in filtered_df.columns:
        filtered_df["score"] -= filtered_df[feature].apply(lambda sentiment: sentiments.get(sentiment, 0))

    return filtered_df

  def _load_from_db(self) -> pd.DataFrame:
    import sqlite3

    db_loc = Path.cwd().joinpath("resources", "database.db")
    with sqlite3.connect(db_loc) as conn:
      query = "SELECT * FROM restaurants"
      df = pd.read_sql_query(query, conn)
    return df


if __name__ == "__main__":
  query = "Recommend a good fast food restaurant in New Orleans"
  recommend_system = RestaurantRecommenter()
  top_restaurants = recommend_system(query)
  print(f"Top restaurants found for {query}")
  print(top_restaurants)
