import json
from dataclasses import asdict
from typing import Any


def serializable(cls: Any) -> Any:
    """
    Decorator to add a `to_json` method to a dataclass.
    The `to_json` method serializes the dataclass instance (including nested dataclasses).
    """
    def to_json(self) -> dict:
        try:
            # Serialize the dataclass instance to a dictionary
            tmp = asdict(self)
            return json.dumps(tmp)
        except Exception as e:
            raise ValueError(f"Serialization failed: {e}")

    setattr(cls, "to_json", to_json)
    return cls