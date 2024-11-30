from dataclasses import dataclass
from interface.response.restaurant import Restaurant
from FastDineAPI.decorators.serializable import serializable

@serializable
@dataclass
class QueryResponse:
  Restaurants: list[Restaurant]