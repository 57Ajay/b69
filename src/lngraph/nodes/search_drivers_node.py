from typing import Dict, Any, List
from src.models.agent_state_model import AgentState
import logging
from langchain_google_vertexai import ChatVertexAI
from src.lngraph.tools.driver_tools import DriverTools
from src.models.drivers_model import DriverModel

logger = logging.getLogger(__name__)


class SearchDriversNode:
    """
    FIXED: Node to handle the driver search intent. It now relies on the state
    being populated by the TripInfoCollectionNode and the router, removing
    redundant and error-prone entity extraction.
    """

    def __init__(self, llm: ChatVertexAI, driver_tools: DriverTools):
        """
        Initializes the SearchDriversNode.

        Args:
            llm: An instance of a language model (kept for potential future use but not for city extraction).
            driver_tools: An instance of the DriverTools class.
        """
        self.llm = llm
        self.driver_tools = driver_tools

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the driver search logic, assuming the city is already present in the state.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary with the updated state values.
        """
        logger.info("Executing SearchDriversNode...")

        # The router now ensures search_city is populated from pickupLocation before calling this node.
        city = state.get("search_city")

        if not city:
            # This should ideally not be reached due to the new router logic, but serves as a safeguard.
            logger.error("SearchDriversNode was called without a city in the state.")
            return {
                "last_error": "I'm missing the city to search in. Could you please clarify your pickup location?",
                "failed_node": "search_drivers_node"
            }

        logger.info(f"Performing a new driver search in city: {city}")

        try:
            # A new search always starts from page 1 and clears previous filters.
            current_page = 1
            tool_response = await self.driver_tools.search_drivers_tool.ainvoke({
                "city": city,
                "page": current_page,
                "limit": state.get("limit", 5),
                "use_cache": True, # Use cache for initial searches
            })

            if tool_response.get("success"):
                drivers: List[DriverModel] = tool_response.get("drivers", [])
                driver_count = len(drivers)

                logger.info(f"Successfully found {driver_count} drivers for the new search.")

                driver_details_for_state = [{"driver_name": driver.name, "driver_id": driver.id} for driver in drivers]

                # CRITICAL: A new search resets the entire driver and filter context.
                return {
                    "search_city": city,
                    "current_page": current_page,
                    "current_drivers": driver_details_for_state,
                    "all_drivers": driver_details_for_state,
                    "total_results": tool_response.get("total", 0),
                    "has_more_results": tool_response.get("has_more", False),
                    "is_filtered": False,
                    "active_filters": {},
                    "selected_driver": None,
                    "driver_summary": None,
                    "last_error": None,
                    "failed_node": None,
                }
            else:
                error_msg = tool_response.get('error', 'An unknown error occurred while searching.')
                logger.error(f"Driver search tool failed: {error_msg}")
                # If the API fails for a valid city, it might be an issue with our service.
                user_friendly_error = f"I'm sorry, I couldn't find any drivers in {city} at the moment. It might be a temporary issue. Please try again shortly."
                return {
                    "last_error": user_friendly_error,
                    "failed_node": "search_drivers_node",
                    "current_drivers": [],
                    "all_drivers": [],
                }
        except Exception as e:
            logger.critical(f"A critical error occurred in SearchDriversNode: {e}", exc_info=True)
            return {
                "last_error": "A system error occurred while searching for drivers. Please try again later.",
                "failed_node": "search_drivers_node",
                "current_drivers": [],
                "all_drivers": [],
            }
