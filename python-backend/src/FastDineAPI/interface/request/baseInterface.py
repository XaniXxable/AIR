from typing import Type, TypeVar, Dict
from dataclasses import dataclass

T = TypeVar('T', bound='BaseInterface')

@dataclass
class BaseInterface:
    @classmethod
    def from_json(cls: Type[T], json_data: Dict) -> T:
        """
        Generic method for deserializing JSON data into a dataclass.
        """
        return cls(**json_data)
