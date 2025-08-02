from typing import TypedDict, Optional, Any, List, Dict
from langchain_core.messages import BaseMessage
from src.models.drivers_model import DriverModel
from src.models.user_model import UserModel


class AgentState(TypedDict):
    # User Session Management
    session_id: str
    user_id: Optional[str]
    user: Optional[UserModel]

    # Conversation State
    messages: List[BaseMessage]
    last_user_message: str
    conversation_language: str  # Auto-detected/switched language

    # Search Context
    pickup_city: Optional[str]
    destination_city: Optional[str]
    current_page: int
    page_size: int
    total_results: int
    has_more_results: bool

    # Active Filters
    active_filters: Dict[str, Any]
    previous_filters: List[Dict[str, Any]]  # Filter history for modifications

    # Driver Context
    current_drivers: List[DriverModel]  # Currently displayed drivers
    selected_driver: Optional[DriverModel]  # Driver user is inquiring about
    driver_history: List[str]  # Driver IDs user has viewed

    # Booking Context
    booking_status: str  # "none", "in_progress", "confirmed"
    booking_details: Optional[Dict[str, Any]]

    # Error Handling
    last_error: Optional[str]
    retry_count: int

    # Cache Keys
    cache_keys_used: List[str]  # For efficient cache management

    # User Preferences (learned over time)
    # Frequently used filters, preferred vehicles
    user_preferences: Dict[str, Any]
