from dataclasses import dataclass
from FastDineAPI.decorators.serializable import serializable

@serializable
@dataclass
class Restaurant:
  Name: str
  Type: str
  Reviews: int
  Location: str
  Image: str