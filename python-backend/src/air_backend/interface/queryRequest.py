from dataclasses import dataclass
from air_backend.interface.restaurant import Restaurant

@dataclass
class QueryResponse:
  restaurants: list[Restaurant]