from typing import Dict, Any, List
from src.models.agent_state_model import AgentState
import logging
from src.lngraph.tools.driver_tools import DriverTools
from src.models.drivers_model import DriverModel

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
                "failed_node": "more_drivers_node",
                "current_drivers": [], # Clear drivers to prevent re-displaying old ones
            }

        # Increment the page number
        current_page = state.get("current_page", 1)
        next_page = current_page + 1

        logger.info(f"Fetching page {next_page} for drivers in {state['search_city']}")

        # Call the search tool with the next page number and active filters
        try:
            tool_response = await self.driver_tools.search_drivers_tool.ainvoke({
                "city": state["search_city"],
                "page": next_page,
                "limit": state["limit"],
                **state.get("active_filters", {})
            })

            if tool_response.get("success"):
                new_drivers: List[DriverModel] = tool_response.get("drivers", [])

                # Combine with existing all_drivers list
                existing_drivers = state.get("all_drivers", [])
                if existing_drivers is None:
                    existing_drivers = []
                updated_all_drivers = existing_drivers + [{"driver_name": driver.name, "driver_id": driver.id} for driver in new_drivers]

                return {
                    "current_page": next_page,
                    "current_drivers": [{"driver_name": driver.name, "driver_id": driver.id} for driver in new_drivers],
                    "all_drivers": updated_all_drivers,
                    "total_results": tool_response.get("total", 0),
                    "has_more_results": tool_response.get("has_more", False),
                    "last_error": None,
                    "failed_node": None,
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
