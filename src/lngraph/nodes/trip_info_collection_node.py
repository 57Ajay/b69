from typing import Dict, Any, Optional
from src.models.agent_state_model import AgentState
import logging
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class TripInfo(BaseModel):
    """
    Pydantic model for extracting trip information from user messages.
    All fields are optional with default None values for Pydantic v2 compatibility.
    """
    pickup_location: Optional[str] = Field(
        default=None,
        description="The pickup location/source city where the trip starts"
    )
    drop_location: Optional[str] = Field(
        default=None,
        description="The drop location/destination city where the trip ends"
    )
    trip_type: Optional[str] = Field(
        default=None,
        description="Type of trip: 'one-way', 'round-trip', or 'multi-city'"
    )
    trip_duration: Optional[int] = Field(
        default=None,
        description="Duration in days for round-trip or multi-city trips"
    )

class TripInfoCollectionNode:
    """
    Node to collect essential trip information before proceeding with driver search.
    """

    def __init__(self, llm: ChatVertexAI):
        """
        Initializes the TripInfoCollectionNode.

        Args:
            llm: An instance of a language model for entity extraction.
        """
        self.llm = llm

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Collects trip information from user message or prompts for missing info.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary with the updated state values.
        """
        logger.info("Executing TripInfoCollectionNode...")

        user_message = state["last_user_message"]

        # Check what trip info we already have
        existing_pickup = state.get("pickupLocation")
        existing_drop = state.get("dropLocation")
        existing_trip_type = state.get("trip_type", "one-way")
        existing_duration = state.get("trip_duration")

        # Extract trip information from the current message
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting trip information from user messages.
            Extract the following information ONLY if explicitly mentioned:
            - pickup_location: Where the trip starts (source city) - only if mentioned
            - drop_location: Where the trip ends (destination city) - only if mentioned
            - trip_type: "one-way", "round-trip", or "multi-city" - only if mentioned
            - trip_duration: Number of days for round-trip or multi-city - only if mentioned

            Examples:
            - "book me a cab from delhi to mumbai" -> pickup: "delhi", drop: "mumbai", type: "one-way"
            - "I need a round trip for 3 days" -> trip_type: "round-trip", duration: 3
            - "from airport to hotel" -> pickup: "airport", drop: "hotel"
            - "delhi" -> pickup: "delhi" (assuming it's pickup location)

            If no information is found for a field, return null for that field."""),
            ("human", "User Message: {user_message}")
        ])

        extract_chain = extract_prompt | self.llm.with_structured_output(TripInfo)

        try:
            raw = await extract_chain.ainvoke({"user_message": user_message})
            trip_info = TripInfo.model_validate(raw)

            # Update state with newly extracted information (only if not None)
            updated_pickup = trip_info.pickup_location if trip_info.pickup_location else existing_pickup
            updated_drop = trip_info.drop_location if trip_info.drop_location else existing_drop
            updated_trip_type = trip_info.trip_type if trip_info.trip_type else existing_trip_type
            updated_duration = trip_info.trip_duration if trip_info.trip_duration else existing_duration

            # Special case: If user just mentions a single city, treat it as pickup
            if not updated_pickup and not updated_drop and trip_info.pickup_location:
                updated_pickup = trip_info.pickup_location

        except Exception as e:
            logger.error(f"Error during trip info extraction: {e}", exc_info=True)
            # Use existing values if extraction fails
            updated_pickup = existing_pickup
            updated_drop = existing_drop
            updated_trip_type = existing_trip_type
            updated_duration = existing_duration

        # Check what's still missing
        missing_info = []
        if not updated_pickup:
            missing_info.append("pickup location")
        if not updated_drop:
            missing_info.append("destination")
        if updated_trip_type == "round-trip" and not updated_duration:
            missing_info.append("trip duration (how many days?)")

        # Determine if we have complete trip information
        has_complete_info = (
            updated_pickup and
            updated_drop and
            updated_trip_type and
            (updated_trip_type != "round-trip" or updated_duration)
        )

        update_dict = {
            "pickupLocation": updated_pickup,
            "dropLocation": updated_drop,
            "trip_type": updated_trip_type,
            "trip_duration": updated_duration,
            "full_trip_details": has_complete_info,
        }

        if has_complete_info:
            logger.info("Complete trip information collected")
            update_dict.update({
                "search_city": updated_pickup,  # Use pickup location as search city
                "last_error": None
            })
        else:
            logger.info(f"Missing trip information: {missing_info}")
            # Create a helpful prompt for missing information
            if len(missing_info) == 1:
                missing_prompt = f"I need to know your {missing_info[0]}. Could you please provide it?"
            else:
                missing_prompt = f"I need a few more details: {', '.join(missing_info[:-1])} and {missing_info[-1]}."

            update_dict.update({
                "last_error": missing_prompt,
                "failed_node": "trip_info_collection"
            })

        return update_dict
