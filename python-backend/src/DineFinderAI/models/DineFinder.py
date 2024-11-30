from interface.request.queryRequest import QueryRequest
from interface.response.queryResponse import QueryResponse

from interface.response.restaurant import Restaurant

class DineFinder:

  def __init__(self) -> None:
    pass

  def execute(self, query: QueryRequest) -> QueryResponse:
    return QueryResponse(
      Restaurants=[
          Restaurant(
              Name="Pasta Palace",
              Type="Italian",
              Reviews=150,
              Location="Main Street",
              Image="pasta.jpg"
          ),
          Restaurant(
              Name="Sushi Spot",
              Type="Japanese",
              Reviews=89,
              Location="Oak Avenue",
              Image="sushi.jpg"
          )
        ]
      )

  def save(self) -> None:
    """
      Save the trained model into models dict.
    """

