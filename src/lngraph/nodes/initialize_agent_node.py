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

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary with minimal state updates.
        """
        logger.info("Executing InitializeAgentNode...")

        # CRITICAL: Only update the last user message, preserve everything else
        updates = {
            "last_user_message": state["messages"][-1].content if state.get("messages") else "",
        }

        logger.debug(f"Preserving state for session {state.get('session_id')}")
        logger.debug(f"Current search_city: {state.get('search_city')}")
        logger.debug(f"Current drivers count: {len(state.get('current_drivers', []))}")

        return updates
