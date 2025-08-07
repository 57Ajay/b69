from typing import Dict, Any, Optional, List
from src.models.drivers_model import DriverModel
from src.models.agent_state_model import AgentState
import logging
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.lngraph.tools.driver_tools import DriverTools

logger = logging.getLogger(__name__)

# --- Pydantic Model for Structured LLM Output ---

class FilterEntities(BaseModel):
    """
    Pydantic model for extracting filter criteria from a user's message.
    Maps directly to the filters available in the driver tools.
    """
    vehicle_types: Optional[List[str]] = Field(None, description="List of vehicle types like 'sedan', 'suv'.")
    gender: Optional[str] = Field(None, description="Driver's gender, e.g., 'male', 'female'.")
    is_pet_allowed: Optional[bool] = Field(None, description="Whether the driver allows pets.")
    min_experience: Optional[int] = Field(None, description="Minimum years of driving experience.")
    min_age: Optional[int] = Field(None, description="Minimum age of the driver.")
    languages: Optional[List[str]] = Field(None, description="List of languages the driver speaks.")
    married: Optional[bool] = Field(None, description="The driver's marital status.")
    clear_previous_filters: Optional[bool] = Field(None, description="Whether to clear all previous filters before applying new ones.")
    allow_handicapped_persons: Optional[bool] = Field(None, description="Whether the driver allows handicapped persons.")
    available_for_customers_personal_car: Optional[bool] = Field(None, description="Whether the driver is available for customers' personal cars.")
    available_for_driving_in_event_wedding: Optional[bool] = Field(None, description="Whether the driver is available for driving in events like weddings.")
    available_for_part_time_full_time: Optional[bool] = Field(None, description="Whether the driver is available for part-time or full-time work.")

class FilterDriversNode:
    """
    FIXED: Node to handle filtering intents with proper API integration and state management.
    This version ensures filters are applied immediately via API calls, not just cached data.
    """

    def __init__(self, llm: ChatVertexAI, driver_tools: DriverTools):
        """
        Initializes the FilterDriversNode.

        Args:
            llm: An instance of a language model for entity extraction.
            driver_tools: An instance of the DriverTools class.
        """
        self.llm = llm
        self.driver_tools = driver_tools

    def _normalize_filter_values(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        FIXED: Normalize filter values to match API expectations.
        """
        normalized = {}

        for key, value in filter_dict.items():
            if key == "vehicle_types" and isinstance(value, list):
                # Convert list to comma-separated string for API
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
                # Keep other filters as-is
                normalized[key] = value

        return normalized

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        FIXED: Executes the logic to extract filters and apply them via fresh API call.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary with the updated state values.
        """
        logger.info("Executing FilterDriversNode...")

        user_message = state["last_user_message"]

        if not state.get("search_city"):
            logger.warning("Filter intent detected without an active search.")
            return {
                "last_error": "It looks like you want to filter, but we haven't searched for any drivers yet. Please tell me which city you're looking in first.",
                "failed_node": "filter_drivers_node"
            }

        # 1. Extract filter criteria from the user's message
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an entity extraction expert. From the user's message, extract any specified filters for a cab driver search.

            Pay special attention to:
            - Vehicle types: sedan, suv, hatchback, luxury, etc.
            - Gender: male, female
            - Pet-friendliness: pets allowed, pet-friendly
            - Experience: years of experience, minimum experience
            - Age: minimum age
            - Languages: hindi, english, punjabi, etc.
            - Marital status: married, single
            - Filter management: "remove old filter", "clear previous", "reset filters"

            If the user wants to clear or remove old filters, set clear_previous_filters to true.
            Only extract the filters the user explicitly mentions in this specific message.

            Examples:
            - "show me sedan drivers" -> vehicle_types: ["sedan"]
            - "married drivers only" -> married: true
            - "drivers who speak hindi" -> languages: ["hindi"]
            - "minimum 5 years experience" -> min_experience: 5
            - "clear filters and show SUV drivers" -> clear_previous_filters: true, vehicle_types: ["suv"]
            """),
            ("human", "User Message: {user_message}")
        ])

        extract_chain = extract_prompt | self.llm.with_structured_output(FilterEntities)

        try:
            raw = await extract_chain.ainvoke({"user_message": user_message})
            extracted_filters = FilterEntities.model_validate(raw)
            filter_dict = extracted_filters.model_dump(exclude_unset=True)
            clear_filters = filter_dict.pop("clear_previous_filters", False)

            if not filter_dict and not clear_filters:
                logger.warning("Filter intent classified, but no specific filters were extracted.")
                return {"last_error": "I understand you want to filter, but I'm not sure what criteria to use. Could you please be more specific? For example: 'show me sedan drivers' or 'married drivers only'"}

        except Exception as e:
            logger.error(f"Error during filter extraction: {e}", exc_info=True)
            return {"last_error": "I had trouble understanding your filter criteria. Could you please rephrase?", "failed_node": "filter_drivers_node"}

        # 2. FIXED: Better filter management
        current_filters = state.get("active_filters", {})

        if clear_filters:
            updated_filters = filter_dict
            logger.info(f"Clearing previous filters and applying new ones: {updated_filters}")
        else:
            # Merge with existing filters, allowing overrides
            updated_filters = {**current_filters, **filter_dict}
            logger.info(f"Merging filters. Previous: {current_filters}, New: {filter_dict}, Combined: {updated_filters}")

        # 3. FIXED: Normalize filters for API call
        normalized_filters = self._normalize_filter_values(updated_filters)
        logger.info(f"Normalized filters for API: {normalized_filters}")

        new_page = state["current_page"]+1
        try:
            # Build API parameters
            api_params = {
                "city": state["search_city"],
                "page": new_page,
                "limit": state["limit"],
                "radius": state.get("radius", 100),
                "search_strategy": state.get("search_strategy", "hybrid"),
                "sort_by": "lastAccess:desc",
                "use_cache": True,
                **normalized_filters
            }

            logger.info(f"Making API call with parameters: {api_params}")

            tool_response = await self.driver_tools.search_drivers_tool.ainvoke(api_params)

            if tool_response.get("success"):
                filtered_drivers: List[DriverModel] = tool_response.get("drivers", [])
                logger.info(f"Successfully filtered drivers. Found {len(filtered_drivers)} matches.")

                current_drivers = [{"driver_name": driver.name, "driver_id": driver.id} for driver in filtered_drivers]

                logger.info(f"Filtered drivers: {current_drivers}")
                all_drivers = state["all_drivers"] if state["all_drivers"] else []
                logger.info(f"Details of page and cirty: -> \npage: {state['current_page']}, city: {state['search_city']}")

                return {
                    "current_drivers": current_drivers,
                    "all_drivers": all_drivers + current_drivers,
                    "active_filters": updated_filters,
                    "last_error": None,
                    "is_filtered": True,
                    "total_filtered_results": len(filtered_drivers),
                    "failed_node": None,
                    "selected_driver": None,
                    "driver_summary": None,
                    "current_page": new_page,
                    "has_more_results": tool_response.get("has_more", False),
                    "total_results": tool_response.get("total", 0)
                }
            else:
                error_msg = tool_response.get('error', 'An unknown error occurred while filtering.')
                logger.error(f"Filter tool failed: {error_msg}")

                # If no results found, provide helpful message
                if "No drivers found" in str(error_msg) or tool_response.get("count", 0) == 0:
                    filter_summary = []
                    for key, value in updated_filters.items():
                        if key == "vehicle_types":
                            filter_summary.append(f"vehicle type: {', '.join(value) if isinstance(value, list) else value}")
                        elif key == "married":
                            filter_summary.append("married drivers" if value else "unmarried drivers")
                        elif key == "min_age":
                            filter_summary.append(f"minimum age {value}")
                        elif key == "min_experience":
                            filter_summary.append(f"minimum {value} years experience")
                        else:
                            filter_summary.append(f"{key.replace('_', ' ')}: {value}")

                    return {
                        "last_error": f"No drivers found in {state['search_city']} matching your criteria: {', '.join(filter_summary)}. Would you like to remove some filters or try different criteria?",
                        "failed_node": "filter_drivers_node",
                        "current_drivers": [],
                        "all_drivers": [],
                        "active_filters": updated_filters,
                        "is_filtered": True
                    }

                return {"last_error": tool_response.get("msg", error_msg), "failed_node": "filter_drivers_node"}

        except Exception as e:
            logger.critical(f"A critical error occurred in FilterDriversNode: {e}", exc_info=True)
            return {"last_error": "A system error occurred while filtering the results.", "failed_node": "filter_drivers_node"}
