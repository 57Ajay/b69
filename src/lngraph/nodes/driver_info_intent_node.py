from typing import Dict, Any, Optional, List
from src.models.agent_state_model import AgentState, DriverDetailsForState
import logging
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.lngraph.tools.driver_tools import DriverTools
from src.models.drivers_model import APIResponse, DriverModel

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

        full_trip_details = state["full_trip_details"]
        pickup = state["pickupLocation"]
        dropoff = state["dropLocation"]
        trip_type = state["trip_type"]
        trip_duration = state["trip_duration"] if trip_type =="round_trip" else None
        if not full_trip_details:
            return {
                "last_error": f"I don't have any trip details to book. Please provide trip details first\
                pickup: {pickup}, dropoff: {dropoff}, trip_type: {trip_type}, and trip_duration: {trip_duration} if trip type is round trip",
                "failed_node": "driver_info_intent_node"
            }


        user_message = state["last_user_message"]

        # Check if we have search_city in state
        if not state.get("search_city"):
            logger.warning("No search city in state for driver info request.")
            return {
                "last_error": "I don't have a search location. Please search for drivers first by specifying a city.",
                "failed_node": "get_driver_info_node"
            }

        # CRITICAL: Get drivers from current_drivers (which includes filtered results)
        # But fallback to all_drivers if current_drivers is empty
        available_drivers = state["current_drivers"] if state["current_drivers"] else []

        if not available_drivers:
            available_drivers = state["all_drivers"] if state["all_drivers"] is not None else []

        if not available_drivers:
            # Try to get from cache as last resort
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

                # Get ALL drivers from cache
                api_response = APIResponse.model_validate(cached_data)
                available_drivers_with_full_information = api_response.data

                for driver in available_drivers_with_full_information:
                    # Process each driver's information here
                    driver_data_for_state = DriverDetailsForState(
                        driver_id=driver.id,
                        driver_name=driver.name,
                    )
                    available_drivers.append(driver_data_for_state)

            except Exception as e:
                logger.error(f"Error retrieving drivers from cache: {e}")
                return {
                    "last_error": "I don't have a list of drivers to choose from. Please perform a search first.",
                    "failed_node": "get_driver_info_node"
                }

        if not available_drivers:
            logger.warning("No drivers available for info request.")
            return {
                "last_error": "I don't have a list of drivers to choose from. Please perform a search first.",
                "failed_node": "get_driver_info_node"
            }

        # 1. Extract which driver the user is asking about
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an entity extraction expert. From the user's message, identify the driver they are asking about. They might use a name, an index (e.g., 'the first one', 'the second driver'), or pronouns referring to a previously discussed driver. Extract either the name or the 1-based index. Return null for fields not mentioned."),
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


        # Fallback: If no identifier extracted, try to find from the current selected driver
        if not target_driver and state.get("selected_driver"):
            target_driver = state["selected_driver"]

        if not target_driver:
            logger.warning(f"Could not find a matching driver for identifier: {identifier.model_dump_json()}")
            available_names = ", ".join([driver["driver_name"] for driver in available_drivers[:5]])  # Show first 5 names
            return {
                "last_error": f"I couldn't find that specific driver. Available drivers include: {available_names}. Please try again.",
                "failed_node": "get_driver_info_node"
            }

        # logger.info(f"Found driver details for: {target_driver['driver_name']}, {target_driver['driver_id']}")

        # 3. Return the complete driver information using the driver model schema
        try:
            full_driver_info = await self.driver_tools.get_driver_info_tool.ainvoke({
                "city": state["search_city"],
                "page": state["current_page"],
                "driverId": target_driver["driver_id"],
            })

            print("\n------------\n")
            logger.info("Success: ", full_driver_info)
            print("\n------------\n")
            if not full_driver_info["success"]:
                return {
                    "last_error": f"Failed to retrieve driver information: {full_driver_info.msg}",
                    "failed_node": "get_driver_info_node"
                }

            driver_info: DriverModel = full_driver_info["driver"]

            vehicle_info = []

            for vehicle in driver_info.verified_vehicles:
                vehicle_detail = f"vehicle_type: {vehicle.vehicle_type}"
                if hasattr(vehicle, "model") and vehicle.model:
                    vehicle_detail += f" vehicle_model:  {vehicle.model}"
                vehicle_info.append(vehicle_detail)
                if hasattr(vehicle, "per_km_cost") and vehicle.per_km_cost:
                    vehicle_detail += f" cost per km: {vehicle.per_km_cost}"
                if hasattr(vehicle, "fuel_type") and vehicle.fuel_type:
                    vehicle_detail += f" fuel_type: {vehicle.fuel_type}"
                if hasattr(vehicle, "reg_no") and vehicle.reg_no:
                    vehicle_detail += f" reg_no: {vehicle.reg_no}"
                if hasattr(vehicle, "images") and vehicle.images:
                    images = [(image.full.url if image.full is not None else "not available") for image in vehicle.images]
                    vehicle_detail += f" images: {images}"
                vehicle_info.append(vehicle_detail)


            # Extract languages properly
            languages =  driver_info.verified_languages

            age = driver_info.age
            driver_summary = {
                "name": driver_info.name,
                "age": age if age is not None and age > 0  else "Not specified",
                "city": driver_info.city,
                "experience": driver_info.experience,  # This should give the correct experience
                "vehicles": vehicle_info,
                "phone": driver_info.phone_no,
                "profile_url": driver_info.constructed_profile_url,
                "languages": languages,
                "pet_allowed": driver_info.is_pet_allowed,
                "connections": driver_info.connections,
                "gender": driver_info.gender,
                "married": driver_info.married,
                "per_km_cost": [cost.per_km_cost for cost in driver_info.verified_vehicles]
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
