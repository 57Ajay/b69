from typing import Dict, Any, Optional, List
from src.models.agent_state_model import AgentState
import logging
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class TripInfo(BaseModel):
    """
    Pydantic model for extracting trip information from user messages.
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
        description="Type of trip: 'one-way', 'round-trip', or 'multi-city'. Default to 'one-way' if not specified but pickup and drop are."
    )
    trip_duration: Optional[int] = Field(
        default=None,
        description="Duration in days for round-trip or multi-city trips"
    )

class TripInfoCollectionNode:
    """
    Node to collect essential trip information before proceeding with driver search.
    FIXED: Better conversation history analysis and persistent state management.
    """

    def __init__(self, llm: ChatVertexAI):
        """
        Initializes the TripInfoCollectionNode.

        Args:
            llm: An instance of a language model for entity extraction.
        """
        self.llm = llm

    def _get_conversation_history(self, messages: List[Any]) -> str:
        """
        Formats the last 10 messages for inclusion in the prompt.
        """
        return "\n".join([
            f"{msg.type}: {msg.content}" for msg in messages[-10:]
        ])

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        FIXED: Collects trip information with better state management, conversation analysis,
        and stricter completion logic.
        """
        logger.info("Executing TripInfoCollectionNode...")

        messages = state.get("messages", [])
        conversation_context = self._get_conversation_history(messages)

        # Get existing trip info from the state
        existing_pickup = state.get("pickupLocation")
        existing_drop = state.get("dropLocation")
        existing_trip_type = state.get("trip_type")
        existing_duration = state.get("trip_duration")

        logger.info(f"Current state before extraction - Pickup: {existing_pickup}, Drop: {existing_drop}, Type: {existing_trip_type}, Duration: {existing_duration}")

        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting trip information from conversations for a cab booking service.
            Analyze the conversation history and the latest user message to extract the following details.
            - pickup_location: The starting city (e.g., "from delhi").
            - drop_location: The destination city (e.g., "to jaipur").
            - trip_type: Must be 'one-way', 'round-trip', or 'multi-city'. If pickup and drop are clear but type isn't, do not assume anything.
            - trip_duration: The number of days for a 'round-trip' or 'multi-city' trip.

            IMPORTANT RULES:
            1.  Prioritize information from the most recent user messages.
            2.  If the user provides two locations like "delhi to jaipur", the first is the pickup and the second is the drop, make sure to understand the context, and if unclear ask from user.
            3.  If a trip type is mentioned, use it. Otherwise, if you have both pickup and drop, but not trip type default to 'none'.
            4.  Only extract trip duration if the trip type is 'round-trip' or 'multi-city'.

            """),
            ("human", "{conversation_context}")
        ])

        extract_chain = extract_prompt | self.llm.with_structured_output(TripInfo)

        updated_pickup = existing_pickup
        updated_drop = existing_drop
        updated_trip_type = existing_trip_type
        updated_duration = existing_duration

        try:
            raw = await extract_chain.ainvoke({"conversation_context": conversation_context})
            trip_info = TripInfo.model_validate(raw)
            logger.info(f"LLM Extracted: {trip_info}")

            # Merge extracted info with existing state, giving priority to newly extracted info
            if trip_info.pickup_location:
                updated_pickup = trip_info.pickup_location
            if trip_info.drop_location:
                updated_drop = trip_info.drop_location
            if trip_info.trip_type:
                updated_trip_type = trip_info.trip_type
            if trip_info.trip_duration:
                updated_duration = trip_info.trip_duration

            # # If we have pickup and drop but no trip type, default to one-way
            # if updated_pickup and updated_drop and not updated_trip_type:
            #     updated_trip_type = "one-way"

        except Exception as e:
            logger.error(f"Error during trip info extraction: {e}", exc_info=True)
            # Proceed with existing state if extraction fails
            pass

        # --- Strict Validation Logic ---
        has_complete_info = False
        missing_prompt = ""

        if not updated_pickup:
            missing_prompt = "I need to know your pickup location to find drivers. Which city are you starting from?"
        elif not updated_drop:
            missing_prompt = f"Okay, starting from {updated_pickup}. Where would you like to go?"
        elif not updated_trip_type:
            missing_prompt = "Is this a one-way trip, a round-trip, or a multi-city journey?"
        elif updated_trip_type in ["round-trip", "multi-city"] and not updated_duration:
            missing_prompt = f"For your {updated_trip_type} from {updated_pickup} to {updated_drop}, how many days will you need the cab?"
        else:
            has_complete_info = True

        logger.info(f"Final state after validation - Pickup: {updated_pickup}, Drop: {updated_drop}, Type: {updated_trip_type}, Duration: {updated_duration}, Complete: {has_complete_info}")

        update_dict = {
            "pickupLocation": updated_pickup,
            "dropLocation": updated_drop,
            "trip_type": updated_trip_type,
            "trip_duration": updated_duration,
            "full_trip_details": has_complete_info,
        }

        if has_complete_info:
            # If info is complete, set the search city and clear any old errors
            update_dict["search_city"] = updated_pickup
            update_dict["last_error"] = None
        else:
            # If info is missing, set an error message to be handled by the response generator
            update_dict["last_error"] = missing_prompt

        return update_dict
