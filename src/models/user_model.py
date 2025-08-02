from pydantic import BaseModel
from typing import List


class UserModel(BaseModel):
    id: str
    username: str
    name: str
    phone_no: str
    preferred_languages: List[str]


__all__ = ["UserModel"]
