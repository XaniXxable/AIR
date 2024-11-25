import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from FastDineAPI.interface.request.queryRequest import QueryRequest
from FastDineAPI.interface.response.queryResponse import QueryResponse
from FastDineAPI.interface.response.restaurant import Restaurant
app = FastAPI()

class MyResponse(BaseModel):
    data: dict

@app.post("/restauratnts/")
async def index(reqeust: Request) -> dict[str, str]:
  try:
    json_string = await reqeust.json()
    query_request = QueryRequest.from_json(json_string)
    print(query_request)
    response = QueryResponse(
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
    return {"Data": response.to_json()}
    # return response.to_json()
  except Exception as err:
    print(err)
    return {"Error": "Parsing error"}


def start() -> None:
  """Launched with `poetry run start` at root level"""
  uvicorn.run("FastDineAPI.app:app", host="0.0.0.0", port=8000, reload=True)
