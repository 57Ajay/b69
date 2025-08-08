from typing import Dict, Any
from src.models.agent_state_model import AgentState
import logging

logger = logging.getLogger(__name__)

class InitializeAgentNode:
    """
    Node to initialize the agent's state for the current turn.
    """

    def __init__(self):
        """
        Initializes the InitializeAgentNode.
        """
        pass

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        CRITICAL FIX: Clears the previous turn's error state to prevent loops
        and ensures the agent starts fresh with the new user message.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary with the minimal state updates for the new turn.
        """
        logger.info("Executing InitializeAgentNode...")

        # Get the latest user message from the history
        last_user_message = ""
        if state.get("messages") and len(state["messages"]) > 0:
            # Ensure we are looking at the last message, which should be from the user
            last_user_message = state["messages"][-1].content

        updates = {
            "last_user_message": last_user_message,
        }

        # If an error was present from the last turn, clear it now that we are
        # processing a new input. This is crucial to prevent the agent from
        # getting stuck on a previous failure.
        if state.get("last_error"):
            updates["last_error"] = ""
            updates["failed_node"] = ""

        logger.debug(f"Initializing turn with user message: '{last_user_message}'")
        return updates
