from dataclasses import dataclass
from interface.request.baseInterface import BaseInterface

@dataclass
class addit_filters(BaseInterface):
  PetFriendly: bool
  FamilyFriendly: bool