"""
Wait For User Input Node for the cab booking agent.
Pauses the graph execution to wait for user response.
"""

from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
import logging

from src.models.agent_state_model import AgentState

logger = logging.getLogger(__name__)


class WaitForUserInputNode:
    """
    Node responsible for:
    1. Pausing graph execution
    2. Setting appropriate state for user input
    3. Maintaining conversation context
    4. Preparing for next user interaction
    """

    def __init__(self):
        """
        Initialize the wait for user input node.
        No LLM needed as this is a control flow node.
        """
        pass

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Prepare state for waiting for user input.

        Args:
            state: Current agent state

        Returns:
            Updated state ready for user input
        """
        try:
            # Log current state summary
            logger.info(
                f"Waiting for user input. Current context: "
                f"City: {state.get('pickup_city')}, "
                f"Drivers shown: {len(state.get('current_drivers', []))}, "
                f"Selected driver: {state.get('selected_driver', {}).get('name') if state.get('selected_driver') else 'None'}"
            )

            # Set conversation state
            state["conversation_state"] = "waiting_for_input"

            # Clear any temporary error states
            if "last_error" in state and state["last_error"] is None:
                del state["last_error"]

            # Reset retry count for next operation
            state["retry_count"] = 0

            # Ensure next node will be entry node for processing user input
            state["next_node"] = "entry_node"

            # Set a flag indicating we're waiting
            state["awaiting_user_input"] = True

            # Store checkpoint for potential recovery
            state["last_stable_state"] = {
                "pickup_city": state.get("pickup_city"),
                "destination_city": state.get("destination_city"),
                "current_page": state.get("current_page", 1),
                "active_filters": state.get("active_filters", {}),
                "selected_driver": state.get("selected_driver"),
            }

            return state

        except Exception as e:
            logger.error(f"Error in wait for user input node: {str(e)}")
            # Even on error, we should wait for user input
            state["next_node"] = "entry_node"
            state["awaiting_user_input"] = True
            return state
