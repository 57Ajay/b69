from typing import Optional, Any, List, Dict, TypedDict
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
    conversation_language: str
    # Added: "waiting_for_input", "ended", etc.
    conversation_state: Optional[str]

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
    # "none", "in_progress", "confirmed", "pending_info", "failed", "error", "completed"
    booking_status: str
    booking_details: Optional[Dict[str, Any]]
    booking_reference: Optional[Dict[str, Any]]  # Added

    # Error Handling
    last_error: Optional[str]
    retry_count: int
    error_history: Optional[List[Dict[str, Any]]]  # Added
    failed_node: Optional[str]  # Added

    # Cache Keys
    cache_keys_used: List[str]  # For efficient cache management

    # User Preferences (learned over time)
    # Frequently used filters, preferred vehicles
    user_preferences: Dict[str, Any]

    # Flow Control - Added fields
    # "search", "filter", "driver_info", "booking", "general_query", "more_results"
    intent: Optional[str]
    next_node: Optional[str]  # Next node to route to
    awaiting_user_input: bool  # Added

    # Additional Context - Added fields
    timestamp: Optional[str]
    created_at: Optional[str]
    last_updated: Optional[str]
    state_version: Optional[str]

    # Recovery and Manual Mode - Added fields
    last_stable_state: Optional[Dict[str, Any]]
    manual_mode: Optional[bool]

    # Suggestions and Options - Added fields
    no_results_suggestions: Optional[List[Dict[str, Any]]]
    suggested_cities: Optional[List[Dict[str, Any]]]
    nearby_city_suggestions: Optional[Dict[str, List[Dict[str, Any]]]]
    suggestion_context: Optional[str]
    quick_city_options: Optional[List[Dict[str, Any]]]
    filter_relaxation_suggestions: Optional[List[Dict[str, Any]]]

    # Search parameters - Added fields
    radius: Optional[int]
    search_strategy: Optional[str]
    use_cache: Optional[bool]

    # Clarification tracking - Added fields
    clarification_attempts: Optional[int]
