from typing import Optional, Any, List, Dict, TypedDict
from langchain_core.messages import BaseMessage
from src.models.drivers_model import DriverModel
from src.models.user_model import UserModel


class AgentState(TypedDict):
    """
    Represents the state of the conversation and agent's memory.
    This TypedDict is used by the LangGraph agent to pass information between nodes.
    """

    # --- User & Session Management ---
    session_id: str
    user: Optional[UserModel]

    # --- Conversation State ---
    messages: List[BaseMessage]
    last_user_message: str
    conversation_language: str
    intent: Optional[str]

    # --- Search Context & Parameters ---
    search_city: Optional[str]
    current_page: int
    page_size: int
    radius: int
    search_strategy: str
    use_cache: bool

    # --- Filtering ---
    active_filters: Dict[str, Any]
    previous_filters: List[Dict[str, Any]]
    is_filtered: bool  # NEW: Track if we're showing filtered results
    total_filtered_results: int  # NEW: Count of filtered results

    # --- API Results & Driver Context ---
    current_drivers: List[DriverModel]  # Currently displayed drivers (may be filtered)
    all_drivers: List[DriverModel]  # NEW: All drivers from original search (preserved)
    total_results: int
    has_more_results: bool
    selected_driver: Optional[DriverModel]
    driver_summary: Optional[Dict[str, Any]]  # Detailed driver info

    # --- Booking Flow ---
    booking_status: str  # e.g., "none", "confirmed", "failed".
    booking_details: Optional[Dict[str, Any]]

    # --- Trip Details ---
    dropLocation: Optional[str]
    pickupLocation: Optional[str]
    trip_type: str
    trip_duration: Optional[int]
    full_trip_details: bool
    trip_doc_id: str

    # --- Error Handling & Flow Control ---
    last_error: Optional[str]
    retry_count: int
    failed_node: Optional[str]
    next_node: Optional[str]

    # --- Suggestions for Enhanced UX ---
    filter_relaxation_suggestions: Optional[List[Dict[str, Any]]]
