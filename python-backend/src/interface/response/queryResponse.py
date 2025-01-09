from dataclasses import dataclass, field, asdict
from interface.response.restaurant import Restaurant
from FastDineAPI.decorators.serializable import serializable


@serializable
@dataclass
class QueryResponse:
  Restaurants: list[Restaurant] = field(default_factory=list)

  def to_dict(self) -> list[dict]:
    return {"Restaurants": [restaurant.to_dict() for restaurant in self.Restaurants]}
