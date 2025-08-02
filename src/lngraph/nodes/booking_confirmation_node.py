"""
Booking Confirmation Node for the cab booking agent.
Handles ride booking and confirmation with selected driver.
"""

from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
import logging

from src.models.agent_state_model import AgentState
from src.models.drivers_model import DriverModel
from src.lngraph.tools.driver_tools import DriverTools

logger = logging.getLogger(__name__)


class BookingConfirmationNode:
    """
    Node responsible for:
    1. Validating booking prerequisites
    2. Confirming driver selection
    3. Checking pickup/destination details
    4. Providing driver contact information
    5. Managing booking flow
    """

    def __init__(self, llm: BaseChatModel, driver_tools: DriverTools):
        """
        Initialize the booking confirmation node.

        Args:
            llm: Language model for generating responses
            driver_tools: Driver tools instance for API calls
        """
        self.llm = llm
        self.driver_tools = driver_tools

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Handle booking confirmation process.

        Args:
            state: Current agent state

        Returns:
            Updated state with booking status
        """
        try:
            # Get selected driver
            selected_driver = state.get("selected_driver")
            if not selected_driver:
                # Try to find driver from context
                error_msg = await self._generate_no_driver_selected_message(
                    state.get("conversation_language", "english")
                )
                state["messages"].append(AIMessage(content=error_msg))
                state["next_node"] = "wait_for_user_input"
                return state

            # Check if we have all required information
            pickup_city = state.get("pickup_city")
            destination_city = state.get("destination_city")

            # Validate booking requirements
            missing_info = []
            if not pickup_city:
                missing_info.append("pickup location")

            # For booking, we might need destination too
            booking_requires_destination = await self._check_if_destination_needed(
                state.get("last_user_message", ""),
                state.get("conversation_language", "english"),
            )

            if booking_requires_destination and not destination_city:
                missing_info.append("destination")

            if missing_info:
                # Ask for missing information
                clarification_msg = await self._generate_missing_info_message(
                    missing_info,
                    selected_driver.name,
                    state.get("conversation_language", "english"),
                )
                state["messages"].append(AIMessage(content=clarification_msg))
                state["booking_status"] = "pending_info"
                state["next_node"] = "wait_for_user_input"
                return state

            # All information available, proceed with booking
            city = pickup_city
            page = state.get("current_page", 1)

            try:
                # Get driver contact details
                result = await self.driver_tools.book_or_confirm_ride_with_driver(
                    city=city, page=page, driverId=selected_driver.id
                )

                if result["success"]:
                    # Update booking status
                    state["booking_status"] = "confirmed"
                    state["booking_details"] = {
                        "driver_id": selected_driver.id,
                        "driver_name": result["Driver Name"],
                        "driver_phone": result["PhoneNo."],
                        "driver_profile": result["Profile"],
                        "pickup_city": pickup_city,
                        "destination_city": destination_city,
                        "vehicle": selected_driver.verified_vehicles[0].model
                        if selected_driver.verified_vehicles
                        else "Not specified",
                    }

                    # Generate confirmation message
                    confirmation_msg = await self._generate_booking_confirmation(
                        state["booking_details"],
                        selected_driver,
                        state.get("conversation_language", "english"),
                    )

                    state["messages"].append(AIMessage(content=confirmation_msg))
                    state["next_node"] = "booking_complete_node"

                else:
                    # Booking failed
                    error_msg = await self._generate_booking_error_message(
                        result.get("msg", "Booking failed"),
                        state.get("conversation_language", "english"),
                    )
                    state["messages"].append(AIMessage(content=error_msg))
                    state["booking_status"] = "failed"
                    state["next_node"] = "wait_for_user_input"

            except Exception as e:
                logger.error(f"Error in booking tool: {e}")
                # Use cached driver info as fallback
                state["booking_status"] = "confirmed"
                state["booking_details"] = {
                    "driver_id": selected_driver.id,
                    "driver_name": selected_driver.name,
                    "driver_phone": selected_driver.phone_no,
                    "driver_profile": selected_driver.constructed_profile_url,
                    "pickup_city": pickup_city,
                    "destination_city": destination_city,
                    "vehicle": selected_driver.verified_vehicles[0].model
                    if selected_driver.verified_vehicles
                    else "Not specified",
                }

                confirmation_msg = await self._generate_booking_confirmation(
                    state["booking_details"],
                    selected_driver,
                    state.get("conversation_language", "english"),
                )

                state["messages"].append(AIMessage(content=confirmation_msg))
                state["next_node"] = "booking_complete_node"

            return state

        except Exception as e:
            logger.error(f"Error in booking confirmation node: {str(e)}")
            state["last_error"] = f"Booking failed: {str(e)}"
            state["booking_status"] = "error"
            state["next_node"] = "error_handler_node"
            return state

    async def _check_if_destination_needed(
        self, user_message: str, language: str
    ) -> bool:
        """
        Check if user's booking request implies a destination is needed.

        Args:
            user_message: User's message
            language: Conversation language

        Returns:
            True if destination needed
        """
        prompt = f"""Analyze if this booking request needs a destination.

User message: "{user_message}"

Booking types:
1. "Book driver for local city travel" - No destination needed
2. "Book ride from X to Y" - Destination needed
3. "Book for daily driving" - No destination needed
4. "Book for trip to Y" - Destination needed

Return only: true or false"""

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip().lower() == "true"
        except:
            # Default to not requiring destination
            return False

    async def _generate_no_driver_selected_message(self, language: str) -> str:
        """
        Generate message when no driver is selected for booking.

        Args:
            language: Response language

        Returns:
            Error message
        """
        prompt = f"""Generate a helpful message when user wants to book but hasn't selected a driver.

Language: {language}

Requirements:
1. Politely explain they need to select a driver first
2. Offer to show available drivers
3. Guide them on how to select (by name or position)
4. Keep conversational and helpful
5. Use {language} language

Generate only the message."""

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return "बुकिंग के लिए पहले एक ड्राइवर चुनें। मैं आपको उपलब्ध ड्राइवर दिखा सकता हूं।"
            else:
                return "Please select a driver first to proceed with booking. Would you like me to show you the available drivers?"

    async def _generate_missing_info_message(
        self, missing_info: list, driver_name: str, language: str
    ) -> str:
        """
        Generate message asking for missing booking information.

        Args:
            missing_info: List of missing information
            driver_name: Selected driver's name
            language: Response language

        Returns:
            Clarification message
        """
        prompt = f"""Generate a message asking for missing booking information.

                    Context:
                        - Selected driver: {driver_name}
                        - Missing information: {", ".join(missing_info)}
                        - Language: {language}

                    Requirements:
                        1. Acknowledge the driver selection
                        2. Ask for the missing information naturally
                        3. If destination missing, ask where they want to go
                        4. If pickup missing, ask where to pick them up
                        5. Keep it conversational
                        6. Use {language} language

                    Generate only the message.
            """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception:
            if language == "hindi":
                return (
                    f"आपने {driver_name} को चुना है। कृपया {', '.join(missing_info)} बताएं।"
                )
            else:
                return f"Great! You've selected {driver_name}. To complete the booking, could you please provide your {' and '.join(missing_info)}?"

    async def _generate_booking_confirmation(
        self, booking_details: Dict[str, Any], driver: DriverModel, language: str
    ) -> str:
        """
        Generate booking confirmation message.

        Args:
            booking_details: Booking details dictionary
            driver: Driver model
            language: Response language

        Returns:
            Confirmation message
        """
        # Prepare additional driver details
        vehicles = [v.model for v in driver.verified_vehicles]
        languages = driver.verified_languages

        prompt = f"""Generate a booking confirmation message.

                    Booking Details:
                        - Driver: {booking_details["driver_name"]}
                        - Phone: {booking_details["driver_phone"]}
                        - Profile: {booking_details["driver_profile"]}
                        - Vehicle: {booking_details["vehicle"]}
                        - Pickup: {booking_details["pickup_city"]}
                        - Destination: {booking_details.get("destination_city", "Not specified")}

                    Additional Info:
                        - Experience: {driver.experience} years
                        - Languages: {", ".join(languages)}
                        - Pet friendly: {"Yes" if driver.is_pet_allowed else "No"}

                    Language: {language}

                    Requirements:
                        1. Confirm the booking clearly
                        2. Show driver contact details prominently
                        3. Mention key driver features
                        4. Provide clear next steps (call driver, view profile)
                        5. Add safety reminder if appropriate
                        6. Thank the user
                        7. Use {language} language
                        8. Format phone number clearly

                    Generate only the confirmation message.
                """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate confirmation: {e}")
            # Fallback
            if language == "hindi":
                return f"✅ बुकिंग कन्फर्म!\n\nड्राइवर: {booking_details['driver_name']}\n📞 फोन: {booking_details['driver_phone']}\n🚗 वाहन: {booking_details['vehicle']}\n\nकृपया ड्राइवर को कॉल करें।"
            else:
                return f"✅ Booking Confirmed!\n\n**Driver Details:**\nName: {booking_details['driver_name']}\n📞 Phone: {booking_details['driver_phone']}\n🚗 Vehicle: {booking_details['vehicle']}\n\nPlease call the driver to coordinate pickup. View full profile: {booking_details['driver_profile']}"

    async def _generate_booking_error_message(self, error: str, language: str) -> str:
        """
        Generate error message for failed booking.

        Args:
            error: Error message
            language: Response language

        Returns:
            User-friendly error message
        """
        prompt = f"""Generate a helpful error message for failed booking.

Error: {error}
Language: {language}

Requirements:
1. Apologize for the issue
2. Suggest trying again or selecting another driver
3. Keep it positive and helpful
4. Use {language} language
5. Don't expose technical details

Generate only the message."""

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return (
                    "क्षमा करें, बुकिंग में समस्या हुई। कृपया दोबारा प्रयास करें या दूसरा ड्राइवर चुनें।"
                )
            else:
                return "I apologize, but I couldn't complete the booking. Please try again or select another driver."
