from interface.response.queryResponse import QueryResponse
from interface.request.queryRequest import QueryRequest
from DineFinderAI.models.DineFinder import DineFinder

model = DineFinder()


def training() -> None:
  request = QueryRequest(None, "Where can I find good bubble tea in philadelphia?")

  data = model.execute(request)
  for restaurant in data:
    print(f"Name: {restaurant['name']}, Score: {restaurant['score']:.2f}")
    print(f"Address: {restaurant['address']}")
    print(f"Categories: {restaurant['categories']}")
    print(f"Stars: {restaurant['stars']}, Reviews: {restaurant['review_count']}")
    print("-" * 50)
