"""
Entry Node for the cab booking agent.
Handles intent classification, entity extraction, and language detection.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
import logging

from src.models.agent_state_model import AgentState

logger = logging.getLogger(__name__)


class EntryNode:
    """
    Entry node responsible for:
    1. Intent classification
    2. Entity extraction (cities, filters, driver references)
    3. Language detection
    4. Initial state setup
    """

    # Intent types
    INTENT_SEARCH = "search"
    INTENT_FILTER = "filter"
    INTENT_DRIVER_INFO = "driver_info"
    INTENT_BOOKING = "booking"
    INTENT_GENERAL = "general_query"
    INTENT_MORE_RESULTS = "more_results"

    def __init__(self, llm: BaseChatModel):
        """
        Initialize the entry node with an LLM for intent classification.

        Args:
            llm: Language model for NLU tasks
        """
        self.llm = llm

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Process user input and classify intent.

        Args:
            state: Current agent state

        Returns:
            Updated state with intent, entities, and language
        """
        try:
            # Get the latest user message
            user_message = state["messages"][-1].content if state["messages"] else ""

            # Skip processing if no user message
            if not user_message:
                return state

            # Update last user message
            state["last_user_message"] = user_message

            # Extract all entities and classify intent using LLM
            extraction_result = await self._extract_entities_and_intent(
                user_message, state
            )

            # Update language if detected
            if extraction_result.get("language"):
                state["conversation_language"] = extraction_result["language"]

            # Extract driver reference separately (needs state context)
            driver_reference = self._extract_driver_reference(user_message, state)
            if driver_reference:
                extraction_result["driver_reference"] = driver_reference

            # Update state with all extracted information
            state = self._update_state_with_entities(
                state,
                extraction_result.get("intent", self.INTENT_GENERAL),
                extraction_result.get("cities", {}),
                extraction_result.get("filters", {}),
                extraction_result.get("driver_reference"),
            )

            # Log the classification
            logger.info(f"Extraction result: {extraction_result}")

            # Set next node based on intent
            state["next_node"] = self._determine_next_node(
                extraction_result.get("intent", self.INTENT_GENERAL), state
            )

            return state

        except Exception as e:
            logger.error(f"Error in entry node: {str(e)}")
            state["last_error"] = f"Failed to process your request: {str(e)}"
            state["retry_count"] = state.get("retry_count", 0) + 1
            return state

    async def _extract_entities_and_intent(
        self, text: str, state: AgentState
    ) -> Dict[str, Any]:
        """
        Extract all entities and classify intent using LLM in a single call.

        Args:
            text: User message
            state: Current agent state

        Returns:
            Dictionary with intent, cities, filters, and language
        """
        has_existing_drivers = bool(state.get("current_drivers"))
        has_selected_driver = bool(state.get("selected_driver"))

        extraction_prompt = f"""Analyze this user message and extract the following information. Return ONLY a JSON object.

User message: "{text}"

Context:
- User has existing driver results: {has_existing_drivers}
- User has a selected driver in context: {has_selected_driver}

Extract:
1. intent: One of ["search", "filter", "driver_info", "booking", "more_results", "general_query"]
   - "search": User wants to find/search for drivers
   - "filter": User wants to apply filters to existing results
   - "driver_info": User wants information about a specific driver
   - "booking": User wants to book/confirm a ride with a driver
   - "more_results": User wants to see more drivers/next page
   - "general_query": Other queries

2. cities: Object with pickup and/or destination cities
   Example: {{"pickup": "delhi", "destination": "mumbai"}}
   - Extract cities from patterns like "from X to Y" or "in X"
   - Validate if cities are in India. If not Indian city, set as null
   - Use lowercase for city names

3. filters: Object with any filter criteria mentioned
   Available filters:
   - vehicle_types: array (sedan, suv, hatchback, innova, innova_crysta, tempo_traveller)
   - gender: string (male/female)
   - min_age, max_age: number
   - is_pet_allowed: boolean
   - min_connections, min_experience: number
   - languages: array of strings
   - profile_verified, married: boolean
   - allow_handicapped_persons: boolean
   - available_for_customers_personal_car: boolean
   - available_for_driving_in_event_wedding: boolean
   - available_for_part_time_full_time: boolean

4. language: Detected language of the message (english, hindi, punjabi, tamil, telugu, etc.)
   - Only include if clearly not English

Return format:
{{
    "intent": "...",
    "cities": {{"pickup": "...", "destination": "..."}},
    "filters": {{...}},
    "language": "..."
}}

Important:
- Set cities to empty object {{}} if no cities mentioned
- Set filters to empty object {{}} if no filters mentioned
- Omit language if English
- For booking/driver_info intents, user must be referring to a specific driver
"""

        try:
            response = await self.llm.ainvoke(extraction_prompt)

            # Parse JSON from response
            import json

            result_text = response.content.strip()

            # Extract JSON from markdown if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            result = json.loads(result_text)

            # Ensure all required fields
            return {
                "intent": result.get("intent", self.INTENT_GENERAL),
                "cities": result.get("cities", {}),
                "filters": result.get("filters", {}),
                "language": result.get("language"),
            }

        except Exception as e:
            logger.warning(f"Failed to extract entities: {e}")
            # Fallback to basic extraction
            return {
                "intent": self.INTENT_GENERAL,
                "cities": {},
                "filters": {},
                "language": None,
            }

    def _extract_driver_reference(self, text: str, state: AgentState) -> Optional[str]:
        """
        Extract driver reference from user message.

        Args:
            text: User message
            state: Current state with driver context

        Returns:
            Driver ID if found, None otherwise
        """
        text_lower = text.lower()
        current_drivers = state.get("current_drivers", [])

        # Check for driver names
        for driver in current_drivers:
            if driver.name.lower() in text_lower:
                return driver.id

        # Check for positional references
        ordinals = {
            "first": 0,
            "1st": 0,
            "second": 1,
            "2nd": 1,
            "third": 2,
            "3rd": 2,
            "fourth": 3,
            "4th": 3,
            "fifth": 4,
            "5th": 4,
            "last": -1,
        }

        for ordinal, index in ordinals.items():
            if ordinal in text_lower and abs(index) <= len(current_drivers):
                return current_drivers[index].id

        # Check for "him/her/this/that driver" with selected driver in context
        context_refs = ["him", "her", "this driver", "that driver", "the driver"]
        if any(ref in text_lower for ref in context_refs) and state.get(
            "selected_driver"
        ):
            return state["selected_driver"].id

        return None

    def _update_state_with_entities(
        self,
        state: AgentState,
        intent: str,
        cities: Dict[str, Optional[str]],
        filters: Dict[str, Any],
        driver_reference: Optional[str],
    ) -> AgentState:
        """
        Update state with extracted entities.

        Args:
            state: Current state
            intent: Classified intent
            cities: Extracted cities
            filters: Extracted filters
            driver_reference: Driver reference

        Returns:
            Updated state
        """
        # Update intent
        state["intent"] = intent

        # Update cities
        if cities.get("pickup"):
            state["pickup_city"] = cities["pickup"]
        if cities.get("destination"):
            state["destination_city"] = cities["destination"]

        # Update filters
        if filters:
            if intent == self.INTENT_FILTER:
                # Merge with existing filters
                state["active_filters"] = {**state.get("active_filters", {}), **filters}
            else:
                # Replace filters for new search
                state["active_filters"] = filters

            # Keep filter history
            if "previous_filters" not in state:
                state["previous_filters"] = []
            state["previous_filters"].append(filters)

        # Update driver reference
        if driver_reference:
            # Find driver in current drivers
            for driver in state.get("current_drivers", []):
                if driver.id == driver_reference:
                    state["selected_driver"] = driver
                    # Add to driver history
                    if "driver_history" not in state:
                        state["driver_history"] = []
                    if driver.id not in state["driver_history"]:
                        state["driver_history"].append(driver.id)
                    break

        return state

    def _determine_next_node(self, intent: str, state: AgentState) -> str:
        """
        Determine the next node based on intent and state.

        Args:
            intent: Classified intent
            state: Current state

        Returns:
            Name of the next node
        """
        # Check if we need city validation first
        if intent in [self.INTENT_SEARCH, self.INTENT_FILTER] and not state.get(
            "pickup_city"
        ):
            return "city_clarification_node"

        # Route based on intent
        intent_to_node = {
            self.INTENT_SEARCH: "driver_search_node",
            self.INTENT_FILTER: "filter_application_node",
            self.INTENT_DRIVER_INFO: "driver_details_node",
            self.INTENT_BOOKING: "booking_confirmation_node",
            self.INTENT_MORE_RESULTS: "pagination_node",
            self.INTENT_GENERAL: "general_response_node",
        }

        return intent_to_node.get(intent, "general_response_node")
