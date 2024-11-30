import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from interface.request.queryRequest import QueryRequest

app = FastAPI()

class MyResponse(BaseModel):
    data: dict

@app.post("/restauratnts/")
async def index(reqeust: Request) -> dict[str, str]:
  try:
    json_string = await reqeust.json()
    query_request = QueryRequest.from_json(json_string)
    # TODO: Call the trianed model.
    return {"Data": ""}
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
