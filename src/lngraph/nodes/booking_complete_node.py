"""
Booking Complete Node for the cab booking agent.
Finalizes the booking process and provides closure.
"""

from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
import logging

from src.models.agent_state_model import AgentState

logger = logging.getLogger(__name__)


class BookingCompleteNode:
    """
    Node responsible for:
    1. Finalizing booking confirmation
    2. Providing booking summary
    3. Offering post-booking assistance
    4. Preparing for conversation end or new search
    """

    def __init__(self, llm: BaseChatModel):
        """
        Initialize the booking complete node.

        Args:
            llm: Language model for generating responses
        """
        self.llm = llm

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Complete the booking process.

        Args:
            state: Current agent state

        Returns:
            Updated state with booking completion
        """
        try:
            booking_details = state.get("booking_details", {})
            language = state.get("conversation_language", "english")

            if not booking_details:
                logger.error("No booking details found in booking complete node")
                state["next_node"] = "error_handler_node"
                return state

            # Generate booking completion message
            completion_message = await self._generate_completion_message(
                booking_details, state, language
            )

            # Add tips or reminders
            tips_message = await self._generate_booking_tips(booking_details, language)

            # Combine messages
            full_message = completion_message
            if tips_message:
                full_message += f"\n\n{tips_message}"

            # Ask if user needs anything else
            followup_message = await self._generate_followup_offer(language)
            full_message += f"\n\n{followup_message}"

            state["messages"].append(AIMessage(content=full_message))

            # Update conversation state
            state["conversation_state"] = "booking_completed"
            state["booking_status"] = "completed"

            # Clear sensitive data but keep booking reference
            state["booking_reference"] = {
                "driver_name": booking_details.get("driver_name"),
                "driver_id": booking_details.get("driver_id"),
                "pickup_city": booking_details.get("pickup_city"),
                "booking_time": state.get("timestamp", ""),
            }

            # Reset search state for potential new booking
            state["current_drivers"] = []
            state["selected_driver"] = None
            state["active_filters"] = {}
            state["current_page"] = 1

            # Log successful booking
            logger.info(
                f"Booking completed for driver: {booking_details.get('driver_name')}"
            )

            # Wait for user to decide next action
            state["next_node"] = "wait_for_user_input"

            return state

        except Exception as e:
            logger.error(f"Error in booking complete node: {str(e)}")
            state["last_error"] = f"Failed to complete booking process: {str(e)}"
            state["next_node"] = "error_handler_node"
            return state

    async def _generate_completion_message(
        self, booking_details: Dict[str, Any], state: AgentState, language: str
    ) -> str:
        """
        Generate booking completion message.

        Args:
            booking_details: Booking information
            state: Current state
            language: Response language

        Returns:
            Completion message
        """
        # Calculate any additional details
        journey_info = ""
        if state.get("destination_city"):
            journey_info = f"Journey: {booking_details['pickup_city']} to {
                state['destination_city']
            }"

        prompt = f"""Generate a booking completion message.

Booking confirmed:
- Driver: {booking_details["driver_name"]}
- Phone: {booking_details["driver_phone"]}
- Vehicle: {booking_details.get("vehicle", "Not specified")}
- {journey_info}

Language: {language}

Requirements:
1. Confirm booking is complete
2. Summarize key details
3. Remind to save driver contact
4. Express good wishes for the journey
5. Be warm and professional
6. Use {language} language
7. Include relevant emojis sparingly

Generate the completion message."""

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate completion message: {e}")
            if language == "hindi":
                return f"✅ बुकिंग पूर्ण!\n\nड्राइवर: {booking_details['driver_name']}\n📞 {booking_details['driver_phone']}\n\nसुरक्षित यात्रा करें!"
            else:
                return f"✅ Booking Complete!\n\nDriver: {booking_details['driver_name']}\n📞 {booking_details['driver_phone']}\n\nHave a safe journey!"

    async def _generate_booking_tips(
        self, booking_details: Dict[str, Any], language: str
    ) -> str:
        """
        Generate helpful tips for the booking.

        Args:
            booking_details: Booking information
            language: Response language

        Returns:
            Tips message
        """
        prompt = f"""Generate 2-3 helpful tips for the confirmed cab booking.

Context:
- Driver booked successfully
- Language: {language}

Tip categories:
- Safety tips
- Communication tips
- Payment tips
- Journey preparation

Requirements:
1. Brief, practical tips
2. Relevant to cab/taxi booking
3. Not patronizing
4. Use {language} language
5. Format as bullet points

Generate only the tips."""

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return "💡 सुझाव:\n• यात्रा विवरण परिवार को बताएं\n• सही किराया पहले तय करें"
            else:
                return "💡 Tips:\n• Share trip details with family\n• Confirm fare before starting"

    async def _generate_followup_offer(self, language: str) -> str:
        """
        Generate followup offer message.

        Args:
            language: Response language

        Returns:
            Followup offer
        """
        prompt = f"""Generate a brief message offering further assistance.

Language: {language}

Options to mention:
- Book another ride
- Search in different city
- Help with anything else

Requirements:
1. Very brief and natural
2. Friendly tone
3. Use {language} language
4. Single sentence preferred

Generate only the followup offer."""

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return "क्या आपको और कुछ चाहिए? मैं एक और राइड बुक कर सकता हूं या अन्य मदद कर सकता हूं।"
            else:
                return "Need anything else? I can help you book another ride or assist with something else."
