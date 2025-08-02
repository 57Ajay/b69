"""
Error Handler Node for the cab booking agent.
Handles system errors, API failures, and unexpected issues gracefully.
"""

from typing import Dict, Any, Optional, List
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
import logging
import traceback

from src.models.agent_state_model import AgentState

logger = logging.getLogger(__name__)


class ErrorHandlerNode:
    """
    Node responsible for:
    1. Handling various types of errors gracefully
    2. Providing user-friendly error messages
    3. Attempting recovery strategies
    4. Logging errors for debugging
    5. Maintaining conversation flow despite errors
    """

    # Error categories and their handling strategies
    ERROR_CATEGORIES = {
        "api_error": {
            "patterns": ["API", "HTTP", "connection", "timeout", "network"],
            "retry": True,
            "user_message": "connection issue with our servers",
        },
        "validation_error": {
            "patterns": ["validation", "invalid", "missing required", "parameter"],
            "retry": False,
            "user_message": "invalid information provided",
        },
        "data_error": {
            "patterns": ["KeyError", "AttributeError", "TypeError", "model_validate"],
            "retry": False,
            "user_message": "data processing issue",
        },
        "cache_error": {
            "patterns": ["cache", "redis", "storage"],
            "retry": True,
            "user_message": "temporary storage issue",
        },
        "tool_error": {
            "patterns": ["tool", "function", "failed to execute"],
            "retry": True,
            "user_message": "service functionality issue",
        },
    }

    def __init__(self, llm: BaseChatModel):
        """
        Initialize the error handler node.

        Args:
            llm: Language model for generating responses
        """
        self.llm = llm

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Handle errors and provide appropriate responses.

        Args:
            state: Current agent state with error information

        Returns:
            Updated state with error handling
        """
        try:
            # Get error information
            last_error = state.get("last_error", "Unknown error")
            retry_count = state.get("retry_count", 0)
            language = state.get("conversation_language", "english")
            failed_node = state.get("failed_node", "unknown")

            # Log the error with full context
            logger.error(f"Error in {failed_node}: {last_error}")
            logger.error(f"Full state keys: {list(state.keys())}")

            # Categorize the error
            error_category = self._categorize_error(last_error)

            # Determine if we should retry
            should_retry = (
                error_category["retry"]
                and retry_count < 3
                and self._is_retryable_context(state)
            )

            # Generate appropriate user message
            if should_retry:
                response = await self._generate_retry_message(
                    error_category, retry_count, state, language
                )
                # Route to error recovery for retry
                state["next_node"] = "error_recovery_node"
                state["retry_count"] = retry_count + 1
            else:
                response = await self._generate_error_message(
                    error_category, state, language
                )
                # Provide alternatives based on context
                alternatives = self._get_alternatives(state)
                if alternatives:
                    alt_message = await self._generate_alternatives_message(
                        alternatives, language
                    )
                    response += f"\n\n{alt_message}"

                # Route to wait for user input
                state["next_node"] = "wait_for_user_input"

                # Reset retry count
                state["retry_count"] = 0

            # Add response to messages
            state["messages"].append(AIMessage(content=response))

            # Store error context for potential recovery
            if "error_history" not in state:
                state["error_history"] = []
            state["error_history"].append(
                {
                    "error": last_error,
                    "node": failed_node,
                    "category": error_category["name"],
                    "timestamp": state.get("timestamp", "unknown"),
                }
            )

            # Clear the error to prevent loops
            state["last_error"] = None

            return state

        except Exception as e:
            # Ultimate fallback - should rarely happen
            logger.critical(f"Error in error handler itself: {str(e)}")
            logger.critical(traceback.format_exc())

            # Provide minimal fallback message
            fallback_msg = self._get_ultimate_fallback(
                state.get("conversation_language", "english")
            )
            state["messages"].append(AIMessage(content=fallback_msg))
            state["next_node"] = "wait_for_user_input"
            state["last_error"] = None

            return state

    def _categorize_error(self, error_message: str) -> Dict[str, Any]:
        """
        Categorize error based on message patterns.

        Args:
            error_message: Error message string

        Returns:
            Error category information
        """
        error_lower = error_message.lower()

        for category_name, category_info in self.ERROR_CATEGORIES.items():
            for pattern in category_info["patterns"]:
                if pattern.lower() in error_lower:
                    return {
                        "name": category_name,
                        "retry": category_info["retry"],
                        "user_message": category_info["user_message"],
                    }

        # Default category
        return {"name": "unknown", "retry": True, "user_message": "technical issue"}

    def _is_retryable_context(self, state: AgentState) -> bool:
        """
        Check if the current context allows retry.

        Args:
            state: Current state

        Returns:
            True if retry is appropriate
        """
        # Don't retry if in certain states
        non_retryable_states = ["booking_complete", "conversation_ended"]
        if state.get("conversation_state") in non_retryable_states:
            return False

        # Don't retry if we've been in error loop
        error_history = state.get("error_history", [])
        if len(error_history) >= 3:
            recent_errors = error_history[-3:]
            if all(
                e["category"] == recent_errors[0]["category"] for e in recent_errors
            ):
                return False

        return True

    def _get_alternatives(self, state: AgentState) -> List[str]:
        """
        Get alternative actions based on current state.

        Args:
            state: Current state

        Returns:
            List of alternative actions
        """
        alternatives = []

        # Based on what user was trying to do
        if state.get("pickup_city"):
            alternatives.append("search in a different city")

        if state.get("active_filters"):
            alternatives.append("search without filters")

        if state.get("current_drivers"):
            alternatives.append("view previously shown drivers")

        if state.get("selected_driver"):
            alternatives.append("view selected driver details")

        # Always available alternatives
        alternatives.extend(["start a new search", "ask for help"])

        return alternatives

    async def _generate_retry_message(
        self,
        error_category: Dict[str, Any],
        retry_count: int,
        state: AgentState,
        language: str,
    ) -> str:
        """
        Generate message for retry attempt.

        Args:
            error_category: Error category info
            retry_count: Number of retries so far
            state: Current state
            language: Response language

        Returns:
            Retry message
        """
        prompt = f"""Generate a brief message about retrying after an error.

                Context:
                    - Error type: {error_category["user_message"]}
                    - Retry attempt: {retry_count + 1}
                    - Language: {language}
                    - User was trying to: {state.get("intent", "search for drivers")}

                Requirements:
                    1. Brief acknowledgment of issue
                    2. Mention we're trying again
                    3. Don't be overly apologetic
                    4. Keep user informed but not worried
                    5. Use {language} language

                Generate only the message.
            """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception:
            if language == "hindi":
                return "तकनीकी समस्या हुई। फिर से कोशिश कर रहे हैं..."
            else:
                return "I encountered a technical issue. Let me try again..."

    async def _generate_error_message(
        self, error_category: Dict[str, Any], state: AgentState, language: str
    ) -> str:
        """
        Generate final error message when not retrying.

        Args:
            error_category: Error category info
            state: Current state
            language: Response language

        Returns:
            Error message
        """
        context = []
        if state.get("intent"):
            context.append(f"User intent: {state['intent']}")
        if state.get("pickup_city"):
            context.append(f"City: {state['pickup_city']}")

        prompt = f"""Generate a helpful error message.

                Context:
                    - Error type: {error_category["user_message"]}
                    - {". ".join(context)}
                    - Language: {language}

                Requirements:
                    1. Acknowledge the issue briefly
                    2. Don't expose technical details
                    3. Stay positive and helpful
                    4. Don't over-apologize
                    5. Focus on what user can do next
                    6. Use {language} language

                Generate only the message.
            """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception:
            if language == "hindi":
                return "कुछ तकनीकी समस्या आई है। कृपया थोड़ी देर बाद प्रयास करें।"
            else:
                return "I'm having some technical difficulties. Please try again in a moment."

    async def _generate_alternatives_message(
        self, alternatives: List[str], language: str
    ) -> str:
        """
        Generate message with alternative actions.

        Args:
            alternatives: List of alternative actions
            language: Response language

        Returns:
            Alternatives message
        """
        prompt = f"""Generate a message suggesting alternative actions.

                Alternatives:
                    {alternatives}

                Language: {language}

                Requirements:
                    1. Present alternatives as helpful suggestions
                    2. Make them actionable
                    3. Keep positive tone
                    4. Use {language} language
                    5. Format clearly (bullets or numbered)

                    Generate only the suggestions."""

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception:
            if language == "hindi":
                return "आप ये कर सकते हैं:\n• नई खोज शुरू करें\n• मदद मांगें"
            else:
                alt_text = "You can:\n"
                for alt in alternatives[:3]:
                    alt_text += f"• {alt.capitalize()}\n"
                return alt_text

    def _get_ultimate_fallback(self, language: str) -> str:
        """
        Get ultimate fallback message when error handler fails.

        Args:
            language: Response language

        Returns:
            Fallback message
        """
        if language == "hindi":
            return "क्षमा करें, कुछ गलत हो गया। कृपया नई खोज शुरू करें या बाद में प्रयास करें।"
        else:
            return "I apologize, something went wrong. Please start a new search or try again later."
