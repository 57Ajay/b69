from typing import Dict, Any
from src.models.agent_state_model import AgentState
import logging

logger = logging.getLogger(__name__)

class InitializeAgentNode:
    """
    Node to initialize the agent's state ONLY if not already initialized.
    """

    def __init__(self):
        """
        Initializes the InitializeAgentNode.
        This node is self-contained and doesn't require external dependencies.
        """
        pass

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        CRITICAL FIX: Only update last_user_message, preserve all other state.
        Ensures that state variables are not reset unexpectedly.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary with minimal state updates.
        """
        logger.info("Executing InitializeAgentNode...")

        # Get the last user message
        last_message = ""
        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1].content

        updates = {
            "last_user_message": last_message,
        }

        if state.get("last_error") and not state.get("search_city"):
            updates["last_error"] = ""
            updates["failed_node"] = ""

        # logger.debug(f"Preserving state for session {state.get('session_id')}")
        # logger.debug(f"Current search_city: {state.get('search_city')}")
        # # logger.debug(f"Current drivers count: {len(state.get('current_drivers', []))}")
        # # logger.debug(f"All drivers count: {len(state.get('all_drivers', []))}")
        # logger.debug(f"Active filters: {state.get('active_filters', {})}")
        # logger.debug(f"Is filtered: {state.get('is_filtered', False)}")

        return updates
