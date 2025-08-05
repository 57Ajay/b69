from typing import Dict, Any, Optional
from src.models.agent_state_model import AgentState
import logging
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.lngraph.tools.driver_tools import DriverTools
from src.models.drivers_model import APIResponse

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
    driver_index: Optional[int] = Field(
        default=None,
        description="The 1-based index of the driver in the list, e.g., 'the first one' -> 1, 'the third driver' -> 3."
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

    def _find_driver_by_name(self, drivers, search_name):
        """
        Find driver by name with fuzzy matching.
        """
        search_name_lower = search_name.lower()

        # First try exact match
        for driver in drivers:
            if driver.name.lower() == search_name_lower:
                return driver

        # Try partial match (driver name contains search term)
        for driver in drivers:
            if search_name_lower in driver.name.lower():
                return driver

        # Try reverse partial match (search term contains driver name parts)
        for driver in drivers:
            driver_name_parts = driver.name.lower().split()
            for part in driver_name_parts:
                if part in search_name_lower and len(part) > 2:  # Avoid matching very short words
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

        # Check if we have search_city in state
        if not state.get("search_city"):
            logger.warning("No search city in state for driver info request.")
            return {
                "last_error": "I don't have a search location. Please search for drivers first by specifying a city.",
                "failed_node": "get_driver_info_node"
            }

        # CRITICAL: Always get ALL drivers from cache, not just filtered ones
        cache_key = self.driver_tools.api_client._generate_cache_key(
            str(state["search_city"]),
            state["current_page"]
        )

        try:
            cached_data = await self.driver_tools.api_client._get_from_cache(cache_key)
            if cached_data is None:
                logger.warning("No drivers in cache to get info for.")
                return {
                    "last_error": "I don't have a list of drivers to choose from. Please perform a search first.",
                    "failed_node": "get_driver_info_node"
                }

            # Get ALL drivers from cache, not just the filtered ones
            api_response = APIResponse.model_validate(cached_data)
            all_drivers = api_response.data

        except Exception as e:
            logger.error(f"Error retrieving drivers from cache: {e}")
            return {
                "last_error": "I don't have a list of drivers to choose from. Please perform a search first.",
                "failed_node": "get_driver_info_node"
            }

        if not all_drivers:
            logger.warning("No drivers in cached data.")
            return {
                "last_error": "I don't have a list of drivers to choose from. Please perform a search first.",
                "failed_node": "get_driver_info_node"
            }

        # 1. Extract which driver the user is asking about
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an entity extraction expert. From the user's message, identify the driver they are asking about. They might use a name or an index (e.g., 'the first one', 'the second driver'). Extract either the name or the 1-based index. Return null for fields not mentioned."),
            ("human", "Available driver names: {driver_names}\n\nUser Message: {user_message}")
        ])

        driver_names = [driver.name for driver in all_drivers]
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
            target_driver = self._find_driver_by_name(all_drivers, identifier.driver_name)
        elif identifier.driver_index and 0 < identifier.driver_index <= len(all_drivers):
            target_driver = all_drivers[identifier.driver_index - 1]

        # Fallback: If no identifier extracted, try to find from the current selected driver
        if not target_driver and state.get("selected_driver"):
            target_driver = state["selected_driver"]

        if not target_driver:
            logger.warning(f"Could not find a matching driver for identifier: {identifier.model_dump_json()}")
            available_names = ", ".join([driver.name for driver in all_drivers[:5]])  # Show first 5 names
            return {
                "last_error": f"I couldn't find that specific driver. Available drivers include: {available_names}. Please try again.",
                "failed_node": "get_driver_info_node"
            }

        logger.info(f"Found driver details for: {target_driver.name}")

        # 3. Return the complete driver information
        try:
            # Extract vehicle information
            vehicle_info = []
            total_cost_per_km = 0
            for vehicle in target_driver.verified_vehicles:
                vehicle_detail = f"{vehicle.vehicle_type}"
                if hasattr(vehicle, 'model') and vehicle.model:
                    vehicle_detail += f" ({vehicle.model})"
                vehicle_info.append(vehicle_detail)
                total_cost_per_km += vehicle.per_km_cost

            avg_cost_per_km = total_cost_per_km / len(target_driver.verified_vehicles) if target_driver.verified_vehicles else 0

            driver_summary = {
                "name": target_driver.name,
                "age": target_driver.age if target_driver.age > 0 else "Not specified",
                "city": target_driver.city,
                "experience": target_driver.experience,
                "vehicles": vehicle_info,
                "avg_cost_per_km": round(avg_cost_per_km, 2),
                "phone": target_driver.phone_no,
                "profile_url": target_driver.constructed_profile_url,
                "languages": target_driver.verified_languages,
                "pet_allowed": target_driver.is_pet_allowed,
                "connections": target_driver.connections
            }

            return {
                "selected_driver": target_driver,
                "driver_summary": driver_summary,
                "last_error": None,
            }
        except Exception as e:
            logger.critical(f"A critical error occurred in GetDriverInfoNode: {e}", exc_info=True)
            return {"last_error": "A system error occurred while fetching driver details.", "failed_node": "get_driver_info_node"}
