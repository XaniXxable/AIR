from dataclasses import asdict
from typing import Any


def serializable(cls: Any) -> Any:
  """
  Decorator to add a `to_dict` method to a dataclass.
  The `to_dict` method serializes the dataclass instance (including nested dataclasses).
  """

  def to_dict(self) -> dict:
    try:
      # Serialize the dataclass instance to a dictionary
      return asdict(self)
    except Exception as e:
      raise ValueError(f"Serialization failed: {e}")

  setattr(cls, "to_dict", to_dict)
  return cls
