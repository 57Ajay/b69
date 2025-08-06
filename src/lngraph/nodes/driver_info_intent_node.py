from typing import Dict, Any, Optional, List
from src.models.agent_state_model import AgentState, DriverDetailsForState
import logging
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.lngraph.tools.driver_tools import DriverTools
from src.models.drivers_model import DriverModel

logger = logging.getLogger(__name__)

# --- Pydantic Model for Structured LLM Output ---

class DriverIdentifier(BaseModel):
    """
    Pydantic model to extract how the user is identifying a driver.
    """
    driver_name: Optional[str] = Field(
        default=None,
        description="The name of the driver the user is asking about, e.g., 'Ramesh'."
    )


class GetDriverInfoNode:
    """
    Node to handle requests for detailed information about a specific driver.
    """

    def __init__(self, llm: ChatVertexAI, driver_tools: DriverTools):
        """
        Initializes the GetDriverInfoNode.

        Args:
            llm: An instance of a language model for entity extraction.
            driver_tools: An instance of the DriverTools class.
        """
        self.llm = llm
        self.driver_tools = driver_tools

    def _find_driver_by_name(self, drivers: List[DriverDetailsForState], search_name: str):
        """
        Find driver by name with fuzzy matching.
        """
        search_name_lower = search_name.lower()

        # First try exact match
        for driver in drivers:
            if driver["driver_name"].lower() == search_name_lower:
                return driver

        # Try partial match (driver name contains search term)
        for driver in drivers:
            if search_name_lower in driver["driver_name"].lower():
                return driver

        # Try reverse partial match (search term contains driver name parts)
        for driver in drivers:
            driver_name_parts = driver["driver_name"].lower().split()
            for part in driver_name_parts:
                if part in search_name_lower and len(part) > 2:
                    return driver

        return None

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the logic to identify a driver and fetch their details.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary with the updated state values.
        """
        logger.info("Executing GetDriverInfoNode...")

        user_message = state["last_user_message"]

        if not state.get("search_city"):
            logger.warning("No search city in state for driver info request.")
            return {
                "last_error": "I don't have a search location. Please search for drivers first by specifying a city.",
                "failed_node": "get_driver_info_node"
            }

        # Use all_drivers for broader context, as current_drivers might be a paginated subset
        available_drivers = state.get("all_drivers", [])

        if not available_drivers:
            logger.warning("No drivers available in the state for info request.")
            return {
                "last_error": "I don't have a list of drivers to choose from. Please perform a search first.",
                "failed_node": "get_driver_info_node"
            }

        # 1. Extract which driver the user is asking about
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an entity extraction expert. From the user's message, identify the driver they are asking about. They might use a name. Extract the name if mentioned."),
            ("human", "Available driver names: {driver_names}\n\nUser Message: {user_message}")
        ])

        driver_names = [driver["driver_name"] for driver in available_drivers]
        extract_chain = extract_prompt | self.llm.with_structured_output(DriverIdentifier)

        try:
            raw = await extract_chain.ainvoke({
                "driver_names": ", ".join(driver_names),
                "user_message": user_message
            })
            identifier = DriverIdentifier.model_validate(raw)
        except Exception as e:
            logger.error(f"Error during driver identification: {e}", exc_info=True)
            return {"last_error": "I'm sorry, I couldn't understand which driver you're asking about.", "failed_node": "get_driver_info_node"}

        # 2. Find the driver from the identifier
        target_driver = None
        if identifier.driver_name:
            target_driver = self._find_driver_by_name(available_drivers, identifier.driver_name)

        if not target_driver and state.get("selected_driver"):
            target_driver = state["selected_driver"]

        if not target_driver:
            available_names = ", ".join([driver["driver_name"] for driver in available_drivers[:5]])
            return {
                "last_error": f"I couldn't find that specific driver. Available drivers include: {available_names}. Please try again.",
                "failed_node": "get_driver_info_node"
            }

        # 3. Fetch full driver details
        try:
            # Determine the page where the driver might be found. This is a simplification;
            # a more robust solution might need to iterate pages or have a better lookup mechanism.
            page_to_check = state.get("current_page", 1)

            full_driver_info = await self.driver_tools.get_driver_info_tool.ainvoke({
                "city": state["search_city"],
                "page": page_to_check,
                "driverId": target_driver["driver_id"],
            })

            if not full_driver_info.get("success"):
                return {
                    "last_error": f"Failed to retrieve driver information: {full_driver_info.get('msg', 'Unknown error')}",
                    "failed_node": "get_driver_info_node"
                }

            driver_info: DriverModel = full_driver_info["driver"]

            vehicle_info = [
                f"vehicle_type: {v.vehicle_type} vehicle_model: {v.model} cost per km: {v.per_km_cost} images: {[img.full.url for img in v.images if img.full]}"
                for v in driver_info.verified_vehicles
            ]

            driver_summary = {
                "name": driver_info.name,
                "age": driver_info.age,
                "city": driver_info.city,
                "experience": driver_info.experience,
                "vehicles": vehicle_info,
                "phone": driver_info.phone_no,
                "profile_url": driver_info.constructed_profile_url,
                "languages": driver_info.verified_languages,
                "pet_allowed": driver_info.is_pet_allowed,
                "married": driver_info.married,
                "gender": driver_info.gender,
                "per_km_cost": [v.per_km_cost for v in driver_info.verified_vehicles],
            }

            return {
                "selected_driver": target_driver,
                "driver_summary": driver_summary,
                "last_error": None,
                "failed_node": None
            }
        except Exception as e:
            logger.critical(f"A critical error occurred in GetDriverInfoNode: {e}", exc_info=True)
            return {"last_error": "A system error occurred while fetching driver details.", "failed_node": "get_driver_info_node"}
