from typing import Dict, Any
from src.models.agent_state_model import AgentState
import logging
from src.lngraph.tools.driver_tools import DriverTools

logger = logging.getLogger(__name__)

class MoreDriversNode:
    """
    Node to handle the 'more_drivers_intent' and fetch the next page of results.
    """

    def __init__(self, driver_tools: DriverTools):
        """
        Initializes the MoreDriversNode.

        Args:
            driver_tools: An instance of the DriverTools class.
        """
        self.driver_tools = driver_tools

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the logic to fetch the next page of drivers.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary with the updated state values.
        """
        logger.info("Executing MoreDriversNode...")

        if not state.get("search_city"):
            return {
                "last_error": "Please tell me which city you're looking in first.",
                "failed_node": "more_drivers_node"
            }

        if not state.get("has_more_results"):
            return {
                "last_error": "I've already shown you all the available drivers.",
                "failed_node": "more_drivers_node"
            }

        # Increment the page number
        current_page = state.get("current_page", 1)
        next_page = current_page + 1

        # Call the search tool with the next page number
        try:
            tool_response = await self.driver_tools.search_drivers_tool.ainvoke({
                "city": state["search_city"],
                "page": next_page,
                "limit": state["limit"],
                **state.get("active_filters", {})
            })

            if tool_response.get("success"):
                return {
                    "current_page": next_page,
                    "total_results": tool_response.get("total", 0),
                    "has_more_results": tool_response.get("has_more", False),
                    "last_error": None,
                }
            else:
                error_msg = tool_response.get('error', 'An unknown error occurred.')
                return {
                    "last_error": error_msg,
                    "failed_node": "more_drivers_node",
                }
        except Exception as e:
            logger.critical(f"A critical error occurred in MoreDriversNode: {e}", exc_info=True)
            return {
                "last_error": "A system error occurred. Please try again later.",
                "failed_node": "more_drivers_node",
            }
