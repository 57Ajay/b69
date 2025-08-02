"""
City Clarification Node for the cab booking agent.
Handles cases where city information is missing and needs to be collected.
"""

from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
import logging

from src.models.agent_state_model import AgentState

logger = logging.getLogger(__name__)


class CityClarificationNode:
    """
    Node responsible for:
    1. Asking for city when missing
    2. Validating city is in India
    3. Handling invalid city inputs
    4. Maintaining conversation flow
    """

    def __init__(self, llm: BaseChatModel):
        """
        Initialize the city clarification node.

        Args:
            llm: Language model for generating responses
        """
        self.llm = llm

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Process city clarification based on context.

        Args:
            state: Current agent state

        Returns:
            Updated state with clarification message
        """
        try:
            # Get conversation language
            language = state.get("conversation_language", "english")

            # Check what's missing
            needs_pickup = not state.get("pickup_city")
            needs_destination = state.get("intent") == "booking" and not state.get(
                "destination_city"
            )

            # Generate appropriate clarification message
            clarification_message = await self._generate_clarification(
                needs_pickup, needs_destination, language, state
            )

            # Add clarification to messages
            state["messages"].append(AIMessage(content=clarification_message))

            # Set next node to entry for re-processing user response
            state["next_node"] = "entry_node"

            # Track this as a clarification attempt
            if "clarification_attempts" not in state:
                state["clarification_attempts"] = 0
            state["clarification_attempts"] += 1

            return state

        except Exception as e:
            logger.error(f"Error in city clarification node: {str(e)}")
            state["last_error"] = f"Failed to process city clarification: {str(e)}"
            state["next_node"] = "error_handler_node"
            return state

    async def _generate_clarification(
        self,
        needs_pickup: bool,
        needs_destination: bool,
        language: str,
        state: AgentState,
    ) -> str:
        """
        Generate appropriate clarification message.

        Args:
            needs_pickup: Whether pickup city is needed
            needs_destination: Whether destination city is needed
            language: Conversation language
            state: Current state for context

        Returns:
            Clarification message
        """
        # Build context for the prompt
        context_parts = []

        # Add what user was trying to do
        intent = state.get("intent", "search")
        if intent == "search":
            context_parts.append("User wants to search for drivers")
        elif intent == "filter":
            context_parts.append("User wants to filter drivers")
        elif intent == "booking":
            context_parts.append("User wants to book a ride")

        # Add any filters they mentioned
        if state.get("active_filters"):
            filters_str = ", ".join(
                [f"{k}: {v}" for k, v in state["active_filters"].items()]
            )
            context_parts.append(f"Requested filters: {filters_str}")

        # Check if this is a repeated clarification
        attempts = state.get("clarification_attempts", 0)

        prompt = f"""Generate a natural clarification message asking for city information.
                    Context:
                        - {". ".join(context_parts)}
                        - Language: {language}
                        - Clarification attempt: {attempts + 1}
                        - Needs pickup city: {needs_pickup}
                        - Needs destination city: {needs_destination}

                    Requirements:
                        1. Ask for the missing city/cities naturally
                        2. Use the same language as the user ({language})
                        3. If this is attempt 2+, acknowledge the repetition politely
                        4. For Indian users, you can mention major cities as examples
                        5. Keep it conversational and helpful
                        6. If user seems to be mentioning a non-Indian city,\
                                politely mention our service is available only in India

                        Generate only the clarification message, nothing else.
                """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate clarification: {e}")
            # Fallback messages based on language
            if language == "hindi":
                if needs_pickup and needs_destination:
                    return "कृपया बताएं कि आप कहाँ से कहाँ जाना चाहते हैं?"
                elif needs_pickup:
                    return "कृपया अपना शहर बताएं जहाँ आपको ड्राइवर चाहिए।"
                else:
                    return "कृपया बताएं कि आप कहाँ जाना चाहते हैं?"
            else:
                if needs_pickup and needs_destination:
                    return (
                        "Could you please tell me your pickup and destination cities?"
                    )
                elif needs_pickup:
                    return "Which city would you like to find drivers in?"
                else:
                    return "Where would you like to go?"
