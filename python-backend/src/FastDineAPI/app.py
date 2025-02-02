import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from interface.request.queryRequest import QueryRequest
from interface.response.queryResponse import QueryResponse
from interface.response.restaurant import Restaurant
from FastDineAPI.recommendation_system.RestaurantRecommender import RestaurantRecommender
import pandas as pd

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:4200"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

recommend_system = RestaurantRecommender()


class MyResponse(BaseModel):
  data: dict


@app.post("/restauratnts/", response_model=QueryResponse)
async def index(reqeust: Request) -> dict[str, str]:
  try:
    json_string = await reqeust.json()
    print(json_string)
    query_request = QueryRequest.from_json(json_string)
    print(f"Processing {query_request.UserInput}")
    feature_weight = None
    top_restaurants: pd.DataFrame = recommend_system(query_request.UserInput, feature_weight)

    restaurants = QueryResponse()
    for _, element in top_restaurants.iterrows():
      restaurant = Restaurant()
      restaurant.Image = "assets/images/restaurant-types/default.jpg"
      restaurant.Location = element["city"]
      restaurant.Type = element["categories"]
      restaurant.Starts = element["stars"]
      restaurant.Score = element["score"]
      restaurant.Name = element["name"]
      restaurant.Reviews = element["review_count"]

      restaurants.Restaurants.append(restaurant)
    return restaurants
  except Exception as err:
    print(err)
    return {"Error": "Parsing error"}


@app.get("/status")
async def status() -> dict[str, str]:
  # TODO: Get the status of the training
  return {"Status": "0%"}


def start() -> None:
  """Launched with `poetry run start` at root level"""
  uvicorn.run("FastDineAPI.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
  start()
