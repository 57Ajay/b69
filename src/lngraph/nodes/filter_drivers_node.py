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
            - "show me sedan and suv drivers" -> vehicle_types: ["sedan", "suv"]
            - "married drivers only" -> married: true
            - "drivers who are married and speak hindi and also have a sedan" -> languages: ["hindi"], vehicle_types: ["sedan"], married: true
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
            # Use .model_dump() with exclude_unset=True to only get explicitly set fields
            filter_dict = extracted_filters.model_dump(exclude_unset=True)

            clear_filters = filter_dict.pop("clear_previous_filters", False)

            if not filter_dict and not clear_filters:
                logger.warning("Filter intent classified, but no specific filters were extracted.")
                unsupported_filters_message = (
                    "I can help with that, but I need to know what to filter by. "
                    "You can filter by vehicle type (SUV, Sedan), driver experience, age, gender, "
                    "languages spoken, marital status, or if they are pet-friendly. "
                    "You can also apply multiple filters at once, like 'show me married SUV drivers'."
                )
                return {"last_error": unsupported_filters_message, "failed_node": "filter_drivers_node"}


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

        # Always start from page 1 for a new filter application
        new_page = state["current_page"] + 1
        try:
            # Build API parameters
            api_params = {
                "city": state["search_city"],
                "page": new_page,
                "limit": state["limit"],
                "use_cache": True,
                **updated_filters
            }

            logger.info(f"Making API call with parameters: {api_params}")

            tool_response = await self.driver_tools.search_drivers_tool.ainvoke(api_params)

            if tool_response.get("success"):
                filtered_drivers: List[DriverModel] = tool_response.get("drivers", [])
                logger.info(f"Successfully filtered drivers. Found {len(filtered_drivers)} matches.")

                # If no drivers are found with the new filters
                if not filtered_drivers:
                    filter_summary_parts = []
                    for key, value in updated_filters.items():
                        filter_summary_parts.append(f"{key.replace('_', ' ')}: {value}")
                    filter_summary = ", ".join(filter_summary_parts)
                    return {
                        "last_error": f"I couldn't find any drivers in {state['search_city']} matching your criteria: {filter_summary}. Would you like to try removing some filters?",
                        "failed_node": "filter_drivers_node",
                        "current_drivers": [],
                        "all_drivers": [], # Reset drivers since the filtered search yielded none
                        "active_filters": updated_filters,
                        "is_filtered": True,
                        "has_more_results": False,
                        "total_results": 0,
                    }


                driver_details_for_state = [{"driver_name": driver.name, "driver_id": driver.id} for driver in filtered_drivers]

                # A new filter action should replace the existing list of drivers
                return {
                    "current_drivers": driver_details_for_state,
                    "all_drivers": driver_details_for_state,
                    "active_filters": updated_filters,
                    "is_filtered": True,
                    "total_filtered_results": tool_response.get("total", 0),
                    "current_page": new_page,
                    "has_more_results": tool_response.get("has_more", False),
                    "total_results": tool_response.get("total", 0),
                    "last_error": None,
                    "failed_node": None,
                    "selected_driver": None,
                    "driver_summary": None,
                }
            else:
                error_msg = tool_response.get('error', 'An unknown error occurred while filtering.')
                logger.error(f"Filter tool failed: {error_msg}")
                return {"last_error": tool_response.get("msg", error_msg), "failed_node": "filter_drivers_node"}

        except Exception as e:
            logger.critical(f"A critical error occurred in FilterDriversNode: {e}", exc_info=True)
            return {"last_error": "A system error occurred while filtering the results.", "failed_node": "filter_drivers_node"}
