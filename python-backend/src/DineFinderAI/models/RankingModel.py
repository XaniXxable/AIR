from DineFinderAI.models.DineFinder import DineFinder


class RankingModel(DineFinder):
  """
    Re-ranking the canditate resulsts from the fine-tuned model.
    It takes the scoring, comments, etc. for the ranking.
  """
  def __init__(self) -> None:
    pass