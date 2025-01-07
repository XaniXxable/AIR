import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from interface.request.queryRequest import QueryRequest
from interface.response.queryResponse import QueryResponse
from interface.response.restaurant import Restaurant
from FastDineAPI.recomentation_system.RestaurantRecommender import RestaurantRecommenter
import pandas as pd
import json

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:4200"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

recommend_system = RestaurantRecommenter()


class MyResponse(BaseModel):
  data: dict


@app.post("/restauratnts/")
async def index(reqeust: Request) -> dict[str, str]:
  try:
    json_string = await reqeust.json()
    print(json_string)
    query_request = QueryRequest.from_json(json_string)
    print(f"Processing {query_request.UserInput}")
    feature_weight = None
    top_restaurants: pd.DataFrame = recommend_system(query_request.UserInput, feature_weight)

    restaurants: list[dict] = []
    for _, element in top_restaurants.iterrows():
      tmp = {}
      tmp["Image"] = "assets/images/restaurant-types/default.jpg"
      tmp["Location"] = element["city"]
      tmp["Type"] = element["categories"]
      tmp["Stars"] = element["stars"]
      tmp["Reviews"] = element["review_count"]

      restaurants.append(tmp)

    response = json.dumps(restaurants)
    print(type(json.dumps(response)))
    # TODO: Call the trianed model.
    return {"Data": response}
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
