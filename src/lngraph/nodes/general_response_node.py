"""
General Response Node for the cab booking agent.
Handles general queries, greetings, and non-specific requests.
"""

from typing import Dict, Any, List, Optional
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
import logging

from src.models.agent_state_model import AgentState

logger = logging.getLogger(__name__)


class GeneralResponseNode:
    """
    Node responsible for:
    1. Handling greetings and general conversation
    2. Answering questions about the service
    3. Providing help and guidance
    4. Managing unclear or ambiguous requests
    """

    def __init__(self, llm: BaseChatModel):
        """
        Initialize the general response node.

        Args:
            llm: Language model for generating responses
        """
        self.llm = llm

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Handle general queries and provide appropriate responses.

        Args:
            state: Current agent state

        Returns:
            Updated state with response
        """
        try:
            user_message = state.get("last_user_message", "")
            language = state.get("conversation_language", "english")

            # Analyze the type of general query
            query_type = await self._analyze_query_type(user_message, language)

            # Generate appropriate response based on query type
            if query_type == "greeting":
                response = await self._generate_greeting_response(state)
            elif query_type == "help":
                response = await self._generate_help_response(state)
            elif query_type == "service_info":
                response = await self._generate_service_info_response(
                    user_message, language
                )
            elif query_type == "status_check":
                response = await self._generate_status_response(state)
            elif query_type == "thanks":
                response = await self._generate_thanks_response(state)
            else:
                # Unclear or ambiguous request
                response = await self._generate_clarification_response(
                    user_message, state
                )

            # Add response to messages
            state["messages"].append(AIMessage(content=response))

            # Always wait for user input after general response
            state["next_node"] = "wait_for_user_input"

            return state

        except Exception as e:
            logger.error(f"Error in general response node: {str(e)}")
            state["last_error"] = f"Failed to generate response: {str(e)}"
            state["next_node"] = "error_handler_node"
            return state

    async def _analyze_query_type(self, message: str, language: str) -> str:
        """
        Analyze the type of general query.

        Args:
            message: User message
            language: Conversation language

        Returns:
            Query type
        """
        prompt = f"""Analyze this general query and categorize it.

                User message: "{message}"
                Language: {language}

                Categories:
                1. "greeting" - Hello, hi, Are you ready to ride?, etc.
                2. "help" - How to use, what can you do, help me
                3. "service_info" - Questions about the service, pricing, areas covered
                4. "status_check" - Where are we, what's happening, current status
                5. "thanks" - Thank you, thanks, appreciate it
                6. "unclear" - Ambiguous or unclear request

                Return only the category name.
        """

        try:
            response = await self.llm.ainvoke(prompt)
            category = response.content.strip().lower()
            if category in [
                "greeting",
                "help",
                "service_info",
                "status_check",
                "thanks",
                "unclear",
            ]:
                return category
            return "unclear"
        except:
            return "unclear"

    async def _generate_greeting_response(self, state: AgentState) -> str:
        """
        Generate greeting response.

        Args:
            state: Current state

        Returns:
            Greeting message
        """
        language = state.get("conversation_language", "english")
        has_previous_interaction = len(state.get("messages", [])) > 1
        user_name = state.get("user", {}).get("name", "")

        prompt = f"""Generate a friendly greeting response.

            Context:
                - Language: {language}
                - Previous interaction: {has_previous_interaction}
                - User name: {user_name if user_name else "Not known"}
                - Service: Cab/driver booking service for Indian cities

            Requirements:
                1. Warm and friendly greeting
                2. If returning user, acknowledge that
                3. Briefly mention what you can help with (finding drivers, booking rides)
                4. Invite them to start searching
                5. Use {language} language
                6. Keep it concise and welcoming

            Generate only the greeting message.
        """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return "नमस्ते! मैं आपको भारत के किसी भी शहर में ड्राइवर खोजने और बुक करने में मदद कर सकता हूं। कहां जाना चाहते हैं?"
            else:
                return "Hello! I can help you find and book drivers in any Indian city. Where would you like to go?"

    async def _generate_help_response(self, state: AgentState) -> str:
        """
        Generate help response explaining capabilities.

        Args:
            state: Current state

        Returns:
            Help message
        """
        language = state.get("conversation_language", "english")

        capabilities = [
            "Search for drivers in any Indian city",
            "Filter by vehicle type, experience, languages, etc.",
            "View driver details and images",
            "Book rides with verified drivers",
            "Find pet-friendly or special service drivers",
        ]

        prompt = f"""Generate a helpful response explaining what the service can do.

                Capabilities:
                    {capabilities}

                Language: {language}

                Requirements:
                    1. Clearly list main features
                    2. Give example commands or queries
                    3. Mention filtering options
                    4. Encourage user to try something
                    5. Use {language} language
                    6. Keep it organized and easy to read

                Generate only the help message.
            """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return "मैं आपकी मदद कर सकता हूं:\n• किसी भी शहर में ड्राइवर खोजें\n• वाहन प्रकार, अनुभव से फिल्टर करें\n• ड्राइवर की जानकारी देखें\n• राइड बुक करें\n\nउदाहरण: 'दिल्ली में SUV ड्राइवर दिखाओ'"
            else:
                return "I can help you:\n• Find drivers in any Indian city\n• Filter by vehicle type, experience, languages\n• View driver details and images\n• Book rides instantly\n\nTry: 'Show me SUV drivers in Delhi' or 'Find pet-friendly drivers in Mumbai'"

    async def _generate_service_info_response(self, message: str, language: str) -> str:
        """
        Generate response about service information.

        Args:
            message: User query
            language: Language

        Returns:
            Service info response
        """
        prompt = f"""Generate a response about our driver booking service.

                User query: "{message}"
                Language: {language}

                Service info:
                    - Coverage: All major Indian cities
                    - Drivers: Verified, experienced professionals
                    - Vehicles: Sedan, SUV, Hatchback, Innova, etc.
                    - Special services: Pet-friendly, handicapped assistance, event driving
                    - Booking: Direct contact with drivers
                    - Available 24/7

                Requirements:
                    1. Answer the specific question if asked
                    2. Provide relevant service information
                    3. Keep it informative but concise
                    4. Use {language} language
                    5. Encourage them to search for drivers

                Generate only the response.
            """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return "हमारी सेवा सभी प्रमुख भारतीय शहरों में उपलब्ध है। सत्यापित ड्राइवर, विभिन्न वाहन, और विशेष सेवाएं उपलब्ध हैं।"
            else:
                return "Our service covers all major Indian cities with verified drivers, various vehicle types, and special services like pet-friendly or event driving. Would you like to search for drivers?"

    async def _generate_status_response(self, state: AgentState) -> str:
        """
        Generate response about current conversation status.

        Args:
            state: Current state

        Returns:
            Status message
        """
        language = state.get("conversation_language", "english")

        # Gather current status
        status_parts = []
        if state.get("pickup_city"):
            status_parts.append(f"City: {state['pickup_city']}")
        if state.get("current_drivers"):
            status_parts.append(f"Found: {len(state['current_drivers'])} drivers")
        if state.get("active_filters"):
            status_parts.append(f"Filters applied: {len(state['active_filters'])}")
        if state.get("selected_driver"):
            status_parts.append(f"Selected: {state['selected_driver'].name}")
        if state.get("booking_status"):
            status_parts.append(f"Booking: {state['booking_status']}")

        prompt = f"""Generate a status update for the user.

                Current status:
                    {status_parts if status_parts else ["No active search or booking"]}

                Language: {language}

                Requirements:
                    1. Summarize current conversation state
                    2. Mention what's been done
                    3. Suggest next possible actions
                    4. Use {language} language
                    5. Keep it clear and helpful

                Generate only the status message.
            """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return "अभी तक कुछ नहीं खोजा गया। आप किस शहर में ड्राइवर खोजना चाहते हैं?"
            else:
                status = (
                    " | ".join(status_parts) if status_parts else "No active search"
                )
                return f"Current status: {status}\n\nWhat would you like to do next?"

    async def _generate_thanks_response(self, state: AgentState) -> str:
        """
        Generate response to thanks.

        Args:
            state: Current state

        Returns:
            Thanks response
        """
        language = state.get("conversation_language", "english")
        has_booking = state.get("booking_status") == "confirmed"

        prompt = f"""Generate a response to user's thanks.

                Context:
                    - Language: {language}
                    - Has completed booking: {has_booking}

                Requirements:
                    1. Acknowledge their thanks warmly
                    2. If they booked, wish them a safe journey
                    3. Offer further assistance
                    4. Use {language} language
                    5. Keep it brief and friendly

                    Generate only the response.
                """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                if has_booking:
                    return "आपका स्वागत है! सुरक्षित यात्रा करें। कुछ और चाहिए तो बताएं।"
                else:
                    return "आपका स्वागत है! कुछ और मदद चाहिए तो बताएं।"
            else:
                if has_booking:
                    return "You're welcome! Have a safe journey. Let me know if you need anything else."
                else:
                    return "You're welcome! Feel free to ask if you need any help."

    async def _generate_clarification_response(
        self, message: str, state: AgentState
    ) -> str:
        """
        Generate response for unclear requests.

        Args:
            message: User message
            state: Current state

        Returns:
            Clarification response
        """
        language = state.get("conversation_language", "english")
        has_context = bool(state.get("current_drivers") or state.get("pickup_city"))

        prompt = f"""Generate a clarification response for an unclear request.

                User message: "{message}"
                Language: {language}
                Has context: {has_context}

                Requirements:
                    1. Politely indicate you didn't fully understand
                    2. If possible, guess what they might want
                    3. Offer specific options they can choose
                    4. If context exists, relate to current state
                    5. Use {language} language
                    6. Keep it helpful and not frustrating

                    Generate only the clarification message.
            """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return "मैं समझ नहीं पाया। क्या आप ड्राइवर खोजना चाहते हैं, फिल्टर लगाना चाहते हैं, या बुकिंग करना चाहते हैं?"
            else:
                return "I'm not sure what you'd like to do. Would you like to:\n• Search for drivers in a city?\n• Filter current results?\n• View driver details?\n• Book a ride?\n\nPlease let me know how I can help!"
