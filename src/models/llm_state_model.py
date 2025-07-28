from pydantic import BaseModel
from models.drivers_model import Driver_Model
from models.user_model import User_Type
from typing import List, Dict, Union, Optional


class User_Journey_Model(BaseModel):
    from_: Optional[str] = None
    to_: Optional[str] = None


class Chat_Model(BaseModel):
    user_chat: str
    Model_response: str


class AGENT_State_Model(BaseModel):
    # user specific stuff
    user: User_Type
    user_journey: Optional[User_Journey_Model] = None
    user_filters: Optional[Dict[str, Union[str | bool]]] = None
    session_id: str

    # LLM STATE SPECIFIC STUFF
    current_drivers: Optional[List[Driver_Model]] = None
    filtered_drivers: Optional[List[Driver_Model]] = None
    chat_history: List[Chat_Model]
    last_user_query: str
    last_model_response: str
    current_step: str = "Intialize agent"

    # SPECIFIC CONFIG
    limit_to_fetch_drivers: int = 10
    current_page: int = 0
    max_page_count: int = 5
