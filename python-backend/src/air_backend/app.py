import uvicorn
from fastapi import FastAPI
from typing import Any

app = FastAPI()

@app.get("/")
def index() -> Any:
  return {"Hello": "World"}


def start() -> None:
  """Launched with `poetry run start` at root level"""
  uvicorn.run("air_backend.app:app", host="0.0.0.0", port=8000, reload=True)
