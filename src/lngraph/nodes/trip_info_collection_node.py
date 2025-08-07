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
    FIXED: Better conversation history analysis and persistent state management.
    """

    def __init__(self, llm: ChatVertexAI):
        """
        Initializes the TripInfoCollectionNode.

        Args:
            llm: An instance of a language model for entity extraction.
        """
        self.llm = llm

    def _analyze_conversation_history(self, messages, current_message: str) -> str:
        """
        Analyze the entire conversation history to extract trip information.
        """
        # Get last 10 messages for context
        recent_messages = messages[-10:] if len(messages) > 10 else messages

        conversation_context = "\n".join([
            f"{msg.type}: {msg.content}" for msg in recent_messages
        ])

        return f"Conversation History:\n{conversation_context}\n\nCurrent Message: {current_message}"

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        FIXED: Collects trip information with better state management and conversation analysis.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary with the updated state values.
        """
        logger.info("Executing TripInfoCollectionNode...")

        user_message = state["last_user_message"]
        messages = state.get("messages", [])

        # Check what trip info we already have in state
        existing_pickup = state.get("pickupLocation")
        existing_drop = state.get("dropLocation")
        existing_trip_type = state.get("trip_type", "")
        existing_duration = state.get("trip_duration")

        logger.info(f"Current state - Pickup: {existing_pickup}, Drop: {existing_drop}, Type: {existing_trip_type}, Duration: {existing_duration}")

        # FIXED: Analyze entire conversation history, not just current message
        conversation_context = self._analyze_conversation_history(messages, user_message)

        # Enhanced extraction prompt that considers conversation history
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting trip information from conversations.
            Analyze the ENTIRE conversation history to extract trip details that may have been mentioned across multiple messages.

            Extract the following information:
            - pickup_location: Where the trip starts (source city) - look for phrases like "from delhi", "pickup from", "starting from"
            - drop_location: Where the trip ends (destination city) - look for phrases like "to mumbai", "going to", "destination"
            - trip_type: "one-way", "round-trip", or "multi-city" - look for keywords like "round trip", "return journey", "back and forth", if not available set None
            - trip_duration: Number of days for round-trip or multi-city trips - look for "3 days", "for 2 days", "week long"

            IMPORTANT:
            1. Look through the entire conversation, not just the current message
            2. If a location is mentioned alone without context, consider it as pickup_location first
            3. Only extract information that is explicitly mentioned

            Examples:
            - "book me a cab from delhi to mumbai" -> pickup: "delhi", drop: "mumbai"
            - "I need drivers from delhi" then "jaipur" -> pickup: "delhi", drop: "jaipur"
            - "delhi to mumbai round trip for 3 days" -> pickup: "delhi", drop: "mumbai", type: "round-trip", duration: 3
            - Just "delhi" -> pickup: "delhi"
            """),
            ("human", "{conversation_context}")
        ])

        extract_chain = extract_prompt | self.llm.with_structured_output(TripInfo)

        try:
            raw = await extract_chain.ainvoke({"conversation_context": conversation_context})
            trip_info = TripInfo.model_validate(raw)

            logger.info(f"Extracted from conversation - Pickup: {trip_info.pickup_location}, Drop: {trip_info.drop_location}, Type: {trip_info.trip_type}, Duration: {trip_info.trip_duration}")

            # FIXED: Better state merging logic
            updated_pickup = trip_info.pickup_location or existing_pickup
            updated_drop = trip_info.drop_location or existing_drop
            updated_trip_type = trip_info.trip_type or existing_trip_type
            updated_duration = trip_info.trip_duration or existing_duration

            # Special handling for single city mentions
            if not updated_pickup and not updated_drop and trip_info.pickup_location:
                updated_pickup = trip_info.pickup_location
            elif updated_pickup and not updated_drop and trip_info.pickup_location and trip_info.pickup_location.lower() != updated_pickup.lower():
                # If we have pickup and user mentions another city, it's likely the destination
                updated_drop = trip_info.pickup_location

        except Exception as e:
            logger.error(f"Error during trip info extraction: {e}", exc_info=True)
            # Use existing values if extraction fails
            updated_pickup = existing_pickup
            updated_drop = existing_drop
            updated_trip_type = existing_trip_type
            updated_duration = existing_duration

        # FIXED: Better missing information detection
        missing_info = []
        if not updated_pickup:
            missing_info.append("pickup location")
        if not updated_drop:
            missing_info.append("destination")
        if not updated_trip_type or updated_trip_type == "":
            missing_info.append("trip type")
        if updated_trip_type in ["round-trip", "multi-city"] and not updated_duration:
            missing_info.append("trip duration (how many days)")

        # Determine if we have complete trip information
        has_complete_info = (
            updated_pickup and
            updated_drop and
            updated_trip_type and
            (updated_trip_type or updated_duration)
        )

        logger.info(f"Final state - Pickup: {updated_pickup}, Drop: {updated_drop}, Type: {updated_trip_type}, Duration: {updated_duration}, Complete: {has_complete_info}")

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
                "search_city": updated_pickup,
                "last_error": None,
                "failed_node": None
            })
        else:
            logger.info(f"Missing trip information: {missing_info}")
            # FIXED: Better prompting for missing information
            if len(missing_info) == 1:
                if "pickup location" in missing_info:
                    missing_prompt = "I need to know where you want to start your trip from. Which city should I search for drivers in?"
                elif "destination" in missing_info:
                    missing_prompt = f"I see you want to start from {updated_pickup}. Where would you like to go? Please tell me your destination city."
                elif "trip duration" in missing_info:
                    missing_prompt = f"For your {updated_trip_type} from {updated_pickup} to {updated_drop}, how many days will you need the cab for?"
                else:
                    missing_prompt = f"I need to know your {missing_info[0]}. Could you please provide it?"
            else:
                missing_prompt = f"I need a few more details to help you: {', '.join(missing_info[:-1])} and {missing_info[-1]}. Please provide these details."

            update_dict.update({
                "last_error": missing_prompt,
                "failed_node": "trip_info_collection"
            })

        return update_dict
