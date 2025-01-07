import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from interface.request.queryRequest import QueryRequest
from interface.response.queryResponse import QueryResponse
from FastDineAPI.recomentation_system import RestaurantRecommender
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


@app.post("/restauratnts/")
async def index(reqeust: Request) -> dict[str, str]:
  try:
    json_string = await reqeust.json()
    query_request = QueryRequest.from_json(json_string)
    print(f"Processing {query_request.UserInput}")
    feature_weight = None
    top_restaurants: pd.DataFrame = recommend_system(query_request.UserInput, feature_weight)
    response = QueryResponse()
    response.Restaurants = top_restaurants.to_dict()
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
