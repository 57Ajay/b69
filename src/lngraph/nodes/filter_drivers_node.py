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
    languages: Optional[List[str]] = Field(None, description="List of languages the driver speaks.")
    married: Optional[bool] = Field(None, description="The driver's marital status.")
    clear_previous_filters: Optional[bool] = Field(None, description="Whether to clear all previous filters before applying new ones.")
    allow_handicapped_persons: Optional[bool] = Field(None, description="Whether the driver allows handicapped persons.")
    available_for_customers_personal_car: Optional[bool] = Field(None, description="Whether the driver is available for customers' personal cars.")
    available_for_driving_in_event_wedding: Optional[bool] = Field(None, description="Whether the driver is available for driving in events like weddings.")
    available_for_part_time_full_time: Optional[bool] = Field(None, description="Whether the driver is available for part-time or full-time work.")

class FilterDriversNode:
    """
    Node to handle filtering intents. It extracts filter criteria, applies them
    to the current search results, and updates the state.
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
        Executes the logic to extract filters and apply them.

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
            - Languages: hindi, english, punjabi, etc.
            - Marital status: married, single
            - Filter management: "remove old filter", "clear previous", "reset filters"

            If the user wants to clear or remove old filters, set clear_previous_filters to true.
            Only extract the filters the user explicitly mentions."""),
            ("human", "User Message: {user_message}")
        ])

        extract_chain = extract_prompt | self.llm.with_structured_output(FilterEntities)

        try:
            raw = await extract_chain.ainvoke({"user_message": user_message})
            extracted_filters = FilterEntities.model_validate(raw)

            # Convert to dict, excluding unset values
            filter_dict = extracted_filters.model_dump(exclude_unset=True)
            clear_filters = filter_dict.pop("clear_previous_filters", False)

            if not filter_dict and not clear_filters:
                logger.warning("Filter intent classified, but no specific filters were extracted.")
                return {"last_error": "I understand you want to filter, but I'm not sure what criteria to use. Could you please be more specific?"}

        except Exception as e:
            logger.error(f"Error during filter extraction: {e}", exc_info=True)
            return {"last_error": "I had trouble understanding your filter criteria. Could you please rephrase?", "failed_node": "filter_drivers_node"}

        # 2. Handle filter management
        current_filters = state.get("active_filters", {})

        if clear_filters:
            # Clear all previous filters and start fresh
            updated_filters = filter_dict
            logger.info(f"Clearing previous filters and applying new ones: {updated_filters}")
        else:
            # Merge with existing filters (new values override old ones)
            updated_filters = {**current_filters, **filter_dict}
            logger.info(f"Merging filters. Previous: {current_filters}, New: {filter_dict}, Combined: {updated_filters}")

        # 3. Call the tool to apply filters to the cached results
        try:
            tool_response = await self.driver_tools.get_drivers_with_user_filter_via_cache_tool.ainvoke({
                "city": state["search_city"],
                "page": state["current_page"],
                "filter_obj": updated_filters,
            })

            logger.info(f"Tool response:----------------------\n \n{tool_response}\n\n---------------------------")

            if tool_response.get("success"):
                filtered_drivers: List[DriverModel] = tool_response.get("filtered_drivers", [])
                logger.info(f"Successfully filtered drivers. Found {len(filtered_drivers)} matches.")

                # CRITICAL: Don't lose the original drivers - keep them accessible
                logger.info("Current Drivers: ",[{"driver_name": driver.name, "driver_id": driver.id} for driver in filtered_drivers])

                return {
                    "current_drivers": [{"driver_name": driver.name, "driver_id": driver.id} for driver in filtered_drivers],
                    "active_filters": updated_filters,
                    "last_error": None,
                    # Keep track that we're in filtered mode
                    "is_filtered": True,
                    "total_filtered_results": len(filtered_drivers),
                    # Clear any previous errors
                    "failed_node": None,
                    # CRITICAL: Clear previous driver selection when filtering
                    "selected_driver": None,
                    "driver_summary": None
                }
            else:
                error_msg = tool_response.get('error', 'An unknown error occurred while filtering.')
                logger.error(f"Filter tool failed: {error_msg}")
                return {"last_error": tool_response.get("msg", error_msg), "failed_node": "filter_drivers_node"}

        except Exception as e:
            logger.critical(f"A critical error occurred in FilterDriversNode: {e}", exc_info=True)
            return {"last_error": "A system error occurred while filtering the results.", "failed_node": "filter_drivers_node"}
