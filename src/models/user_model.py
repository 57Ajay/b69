from pydantic import BaseModel
from typing import List


class User_Type(BaseModel):
    id: str
    username: str
    name: str
    phone_no: str
    preferred_languages: List[str]


__all__ = ["User_Type"]
