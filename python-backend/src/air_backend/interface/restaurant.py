from dataclasses import dataclass

@dataclass
class Restaurant:
  name: str
  category: str
  reviews: int
  location: str
  image: str