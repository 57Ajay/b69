from typing import Dict, Any
from src.models.agent_state_model import AgentState
import logging

logger = logging.getLogger(__name__)

class InitializeAgentNode:
    """
    Node to initialize the agent's state at the beginning of a conversation.
    """

    def __init__(self):
        """
        Initializes the InitializeAgentNode.
        This node is self-contained and doesn't require external dependencies.
        """
        pass

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the node's logic to set up the initial agent state.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary with the updated state values.
        """
        logger.info("Executing InitializeAgentNode...")

        # Default values for a new session
        initialized_state = {
            "user": state.get("user"),
            "last_user_message": state["messages"][-1].content if state.get("messages") else "",
            "conversation_language": "en",
            "intent": None,
            "search_city": None,
            "current_page": 1,
            "page_size": 10,
            "radius": 100,
            "search_strategy": "hybrid",
            "use_cache": True,
            "active_filters": {},
            "previous_filters": [],
            "current_drivers": [],
            "total_results": 0,
            "has_more_results": False,
            "selected_driver": None,
            "booking_status": "none",
            "booking_details": None,
            "dropLocation": None,
            "pickupLocation": None,
            "trip_type": "one-way", # Default trip type
            "trip_duration": None,
            "full_trip_details": False,
            "trip_doc_id": "",
            "last_error": None,
            "retry_count": 0,
            "failed_node": None,
            "next_node": None,
            "filter_relaxation_suggestions": None,
        }

        updated_state = {**state, **initialized_state}
        logger.debug(f"Initialized state for session {updated_state.get('session_id')}")

        return updated_state
