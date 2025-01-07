from dataclasses import dataclass
from FastDineAPI.decorators.serializable import serializable


@serializable
@dataclass
class Restaurant:
  Name: str | None = None
  Type: str | None = None
  Reviews: int | None = None
  Location: str | None = None
  Image: str | None = None
