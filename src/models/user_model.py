from pydantic import BaseModel
from typing import List


class UserModel(BaseModel):
    id: str
    name: str
    phone_no: str
    profile_image: str
    preferred_languages: List[str] = ["english"]


__all__ = ["UserModel"]
