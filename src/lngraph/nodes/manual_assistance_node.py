"""
Manual Assistance Node for the cab booking agent.
Provides manual instructions when automated processes fail.
"""

from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
import logging

from src.models.agent_state_model import AgentState

logger = logging.getLogger(__name__)


class ManualAssistanceNode:
    """
    Node responsible for:
    1. Providing manual booking instructions
    2. Giving direct contact information
    3. Offering alternative booking methods
    4. Handling cases where automation fails
    """

    def __init__(self, llm: BaseChatModel):
        """
        Initialize the manual assistance node.

        Args:
            llm: Language model for generating responses
        """
        self.llm = llm

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Provide manual assistance based on context.

        Args:
            state: Current agent state

        Returns:
            Updated state with manual instructions
        """
        try:
            language = state.get("conversation_language", "english")
            context = self._determine_assistance_context(state)

            # Generate appropriate manual assistance
            if context == "booking_help":
                response = await self._generate_manual_booking_help(state, language)
            elif context == "contact_help":
                response = await self._generate_contact_help(state, language)
            elif context == "search_help":
                response = await self._generate_search_help(state, language)
            else:
                response = await self._generate_general_help(state, language)

            # Add response to messages
            state["messages"].append(AIMessage(content=response))

            # Set manual mode flag
            state["manual_mode"] = True

            # Always wait for user input after manual assistance
            state["next_node"] = "wait_for_user_input"

            return state

        except Exception as e:
            logger.error(f"Error in manual assistance node: {str(e)}")
            # Provide basic fallback assistance
            fallback = self._get_fallback_assistance(
                state.get("conversation_language", "english")
            )
            state["messages"].append(AIMessage(content=fallback))
            state["next_node"] = "wait_for_user_input"
            return state

    def _determine_assistance_context(self, state: AgentState) -> str:
        """
        Determine what type of manual assistance is needed.

        Args:
            state: Current state

        Returns:
            Assistance context type
        """
        # Check various state indicators
        if state.get("booking_status") == "pending_info" or state.get(
            "selected_driver"
        ):
            return "booking_help"
        elif state.get("current_drivers") and len(state["current_drivers"]) > 0:
            return "contact_help"
        elif state.get("pickup_city"):
            return "search_help"
        else:
            return "general_help"

    async def _generate_manual_booking_help(
        self, state: AgentState, language: str
    ) -> str:
        """
        Generate manual booking instructions.

        Args:
            state: Current state
            language: Response language

        Returns:
            Manual booking help message
        """
        driver = state.get("selected_driver")
        driver_info = {}

        if driver:
            driver_info = {
                "name": driver.name,
                "phone": driver.phone_no,
                "vehicles": [v.model for v in driver.verified_vehicles],
                "languages": driver.verified_languages,
            }

        prompt = f"""Generate manual booking instructions.

Context:
- Selected driver info: {driver_info}
- Language: {language}
- User needs help completing booking manually

Requirements:
1. Provide clear step-by-step instructions
2. Include driver contact details prominently
3. Suggest what to discuss when calling
4. Mention important booking details to confirm
5. Be helpful and encouraging
6. Use {language} language

Generate the complete manual booking assistance message."""

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate manual booking help: {e}")
            if language == "hindi":
                return f"बुकिंग के लिए:\n1. ड्राइवर को कॉल करें: {driver_info.get('phone', 'N/A')}\n2. अपना पिकअप स्थान बताएं\n3. किराया तय करें\n4. समय की पुष्टि करें"
            else:
                return f"To complete booking:\n1. Call the driver at: {driver_info.get('phone', 'N/A')}\n2. Provide your pickup location\n3. Confirm the fare\n4. Agree on pickup time"

    async def _generate_contact_help(self, state: AgentState, language: str) -> str:
        """
        Generate help for contacting drivers directly.

        Args:
            state: Current state
            language: Response language

        Returns:
            Contact help message
        """
        drivers = state.get("current_drivers", [])[:3]  # Top 3 drivers

        driver_contacts = []
        for driver in drivers:
            driver_contacts.append(
                {
                    "name": driver.name,
                    "phone": driver.phone_no,
                    "vehicle": driver.verified_vehicles[0].model
                    if driver.verified_vehicles
                    else "N/A",
                }
            )

        prompt = f"""Generate instructions for contacting drivers directly.

Available drivers:
{driver_contacts}

Language: {language}

Requirements:
1. List driver contacts clearly
2. Provide tips for calling drivers
3. Suggest questions to ask
4. Mention negotiation tips
5. Be practical and helpful
6. Use {language} language

Generate the contact assistance message."""

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return "ड्राइवर से सीधे संपर्क करें। फोन करते समय अपना स्थान, समय और किराया स्पष्ट रूप से बताएं।"
            else:
                return "Contact drivers directly. When calling, clearly state your location, timing, and discuss the fare upfront."

    async def _generate_search_help(self, state: AgentState, language: str) -> str:
        """
        Generate manual search instructions.

        Args:
            state: Current state
            language: Response language

        Returns:
            Search help message
        """
        city = state.get("pickup_city", "your city")

        prompt = f"""Generate manual search tips.

Context:
- Searching in: {city}
- Automated search having issues
- Language: {language}

Requirements:
1. Suggest alternative search methods
2. Provide tips for finding drivers
3. Mention peak times for availability
4. Suggest local alternatives
5. Keep it practical
6. Use {language} language

Generate the search assistance message."""

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return f"{city} में ड्राइवर खोजने के लिए:\n• विभिन्न समय पर खोजें\n• आस-पास के शहर देखें\n• फिल्टर हटाएं\n• बाद में फिर कोशिश करें"
            else:
                return f"To find drivers in {city}:\n• Try searching at different times\n• Check nearby cities\n• Remove filters\n• Try again later"

    async def _generate_general_help(self, state: AgentState, language: str) -> str:
        """
        Generate general help instructions.

        Args:
            state: Current state
            language: Response language

        Returns:
            General help message
        """
        prompt = f"""Generate general help for using the cab booking service.

Language: {language}

Requirements:
1. Explain basic usage steps
2. Provide troubleshooting tips
3. Mention common issues and solutions
4. Give contact/support information if available
5. Be comprehensive but clear
6. Use {language} language

Generate the general help message."""

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return "मदद:\n1. अपना शहर बताएं\n2. ड्राइवर खोजें\n3. फिल्टर लगाएं (वैकल्पिक)\n4. ड्राइवर चुनें\n5. बुकिंग करें\n\nसमस्या होने पर नई खोज शुरू करें।"
            else:
                return "How to use:\n1. Tell me your city\n2. Search for drivers\n3. Apply filters (optional)\n4. Select a driver\n5. Complete booking\n\nIf you face issues, try starting a new search."

    def _get_fallback_assistance(self, language: str) -> str:
        """
        Get fallback assistance message.

        Args:
            language: Response language

        Returns:
            Fallback assistance
        """
        if language == "hindi":
            return "तकनीकी समस्या के कारण स्वचालित सहायता उपलब्ध नहीं है। कृपया ड्राइवर से सीधे संपर्क करें या बाद में प्रयास करें।"
        else:
            return "Due to technical issues, automated assistance is unavailable. Please contact drivers directly or try again later."
