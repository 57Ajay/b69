from typing import Dict, Any, List
from src.models.agent_state_model import AgentState
import logging
from src.lngraph.tools.driver_tools import DriverTools
from src.models.drivers_model import DriverModel

logger = logging.getLogger(__name__)

class MoreDriversNode:
    """
    FIXED: Node to handle the 'more_drivers_intent' with proper filter persistence.
    This version ensures that active filters are maintained when fetching more drivers.
    """

    def __init__(self, driver_tools: DriverTools):
        """
        Initializes the MoreDriversNode.

        Args:
            driver_tools: An instance of the DriverTools class.
        """
        self.driver_tools = driver_tools

    def _normalize_filters_for_api(self, active_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        FIXED: Convert user-friendly filters to API format.
        This now handles all supported filters correctly and is consistent
        with the FilterDriversNode.
        """
        normalized = {}

        for key, value in active_filters.items():
            if value is None:
                continue
            if key == "vehicle_types" and isinstance(value, list):
                normalized["vehicleTypes"] = ",".join(value)
            elif key == "languages" and isinstance(value, list):
                normalized["verifiedLanguages"] = ",".join(value)
            elif key == "is_pet_allowed" and isinstance(value, bool):
                normalized["isPetAllowed"] = str(value).lower()
            elif key == "min_experience":
                normalized["minExperience"] = value
            elif key == "min_age":
                normalized["minAge"] = value
            elif key == "gender":
                normalized["gender"] = value
            elif key == "married" and isinstance(value, bool):
                normalized["married"] = value
            elif key == "allow_handicapped_persons":
                normalized["allowHandicappedPersons"] = value
            elif key == "available_for_customers_personal_car":
                normalized["availableForCustomersPersonalCar"] = value
            elif key == "available_for_driving_in_event_wedding":
                normalized["availableForDrivingInEventWedding"] = value
            elif key == "available_for_part_time_full_time":
                normalized["availableForPartTimeFullTime"] = value
            else:
                normalized[key] = value

        return normalized


    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        FIXED: Executes the logic to fetch the next page of drivers with active filters.

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

        if not state.get("has_more_results", False):
            # This is not an error, but a final state. The response generator will handle the message.
            return {
                "messages": ["I've already shown you all the available drivers for your search criteria."],
            }


        current_page = state.get("current_page", 1)
        next_page = current_page + 1

        active_filters = state.get("active_filters", {})
        normalized_filters = self._normalize_filters_for_api(active_filters)

        logger.info(f"Fetching page {next_page} for drivers in {state['search_city']} with filters: {active_filters}")

        api_params = {
            "city": state["search_city"],
            "page": next_page,
            "limit": state["limit"],
            **normalized_filters
        }

        logger.info(f"API call parameters: {api_params}")

        try:
            tool_response = await self.driver_tools.search_drivers_tool.ainvoke(api_params)

            if tool_response.get("success"):
                new_drivers: List[DriverModel] = tool_response.get("drivers", [])

                if not new_drivers:
                    # This can happen if the last page was full but there are no more results
                    return {
                        "has_more_results": False,
                        "messages": ["No more drivers found matching your criteria."],
                    }

                existing_all_drivers = state.get("all_drivers", [])
                if existing_all_drivers is None:
                    existing_all_drivers = []

                new_driver_entries = [{"driver_name": driver.name, "driver_id": driver.id} for driver in new_drivers]
                updated_all_drivers = existing_all_drivers + new_driver_entries

                logger.info(f"Successfully fetched {len(new_drivers)} additional drivers. Total now: {len(updated_all_drivers)}")

                return {
                    "current_page": next_page,
                    "current_drivers": new_driver_entries,
                    "all_drivers": updated_all_drivers,
                    "has_more_results": tool_response.get("has_more", False),
                    "last_error": None,
                    "failed_node": None,
                }
            else:
                error_msg = tool_response.get('error', 'An unknown error occurred.')
                logger.error(f"More drivers API call failed: {error_msg}")
                return {
                    "last_error": f"Failed to fetch more drivers: {error_msg}",
                    "failed_node": "more_drivers_node",
                }
        except Exception as e:
            logger.critical(f"A critical error occurred in MoreDriversNode: {e}", exc_info=True)
            return {
                "last_error": "A system error occurred. Please try again later.",
                "failed_node": "more_drivers_node",
            }
