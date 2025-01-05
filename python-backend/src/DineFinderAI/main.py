from interface.response.queryResponse import QueryResponse
from interface.request.queryRequest import QueryRequest
from DineFinderAI.models.DineFinder import DineFinder
from DineFinderAI.models.FineTuning import main, precompute_restaurant_embeddings, get_top_responses_for_query

# model = DineFinder()


def training() -> None:
  # request = QueryRequest(None, "Where can I find good bubble tea in philadelphia?")

  # data = model.execute(request)
  # for restaurant in data:
  #   print(f"Name: {restaurant['name']}, Score: {restaurant['score']:.2f}")
  #   print(f"Address: {restaurant['address']}")
  #   print(f"Categories: {restaurant['categories']}")
  #   print(f"Stars: {restaurant['stars']}, Reviews: {restaurant['review_count']}")
  #   print("-" * 50)
  # main()
  pass


# precompute_restaurant_embeddings()
data = get_top_responses_for_query("Where can I find good bubble tea in philadelphia?")
for restaurant in data:
  print(f"Name: {restaurant['name']}, Score: {restaurant['score']:.2f}")
  print(f"Address: {restaurant['address']}")
  print(f"Categories: {restaurant['categories']}")
  print("-" * 50)


# training()
