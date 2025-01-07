from typing import Type
from dataclasses import dataclass
from interface.request.baseInterface import BaseInterface
from interface.request.filters import addit_filters


@dataclass
class QueryRequest(BaseInterface):
  Filters: list[addit_filters] | None
  UserInput: str

  @classmethod
  def from_json(cls: Type["QueryRequest"], data: dict) -> "QueryRequest":
    return cls(
      Filters=[addit_filters.from_json(filter_data) for filter_data in data["Filters"]], UserInput=data["UserInput"]
    )
