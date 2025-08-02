"""
Error Recovery Node for the cab booking agent.
Attempts to recover from errors by retrying failed operations intelligently.
"""

from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
import logging
import asyncio

from src.models.agent_state_model import AgentState

logger = logging.getLogger(__name__)


class ErrorRecoveryNode:
    """
    Node responsible for:
    1. Attempting to retry failed operations
    2. Implementing backoff strategies
    3. Trying alternative approaches
    4. Graceful degradation when recovery fails
    """

    # Recovery strategies for different scenarios
    RECOVERY_STRATEGIES = {
        "driver_search_node": {
            "primary": "retry_with_backoff",
            "fallback": "search_without_filters",
            "final": "suggest_different_city",
        },
        "filter_application_node": {
            "primary": "retry_from_cache",
            "fallback": "fresh_search",
            "final": "remove_problematic_filters",
        },
        "booking_confirmation_node": {
            "primary": "retry_with_cached_data",
            "fallback": "show_contact_only",
            "final": "manual_booking_instructions",
        },
        "driver_details_node": {
            "primary": "use_cached_driver",
            "fallback": "basic_info_only",
            "final": "suggest_different_driver",
        },
    }

    def __init__(self, llm: BaseChatModel):
        """
        Initialize the error recovery node.

        Args:
            llm: Language model for generating responses
        """
        self.llm = llm

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Attempt to recover from the error.

        Args:
            state: Current agent state with error information

        Returns:
            Updated state after recovery attempt
        """
        try:
            # Get error context
            failed_node = state.get("failed_node", "unknown")
            retry_count = state.get("retry_count", 1)
            last_error = state.get("last_error", "")
            language = state.get("conversation_language", "english")

            logger.info(f"Attempting recovery for {failed_node}, attempt {retry_count}")

            # Get recovery strategy
            strategy = self.RECOVERY_STRATEGIES.get(
                failed_node,
                {
                    "primary": "generic_retry",
                    "fallback": "skip_operation",
                    "final": "user_guidance",
                },
            )

            # Determine which strategy level to use based on retry count
            if retry_count == 1:
                recovery_method = strategy["primary"]
            elif retry_count == 2:
                recovery_method = strategy["fallback"]
            else:
                recovery_method = strategy["final"]

            # Add exponential backoff for retries
            if retry_count > 1:
                backoff_time = min(2 ** (retry_count - 1), 8)  # Max 8 seconds
                logger.info(f"Applying backoff of {backoff_time} seconds")
                await asyncio.sleep(backoff_time)

            # Execute recovery based on method
            recovery_result = await self._execute_recovery(
                recovery_method, state, failed_node
            )

            if recovery_result["success"]:
                # Recovery successful
                success_msg = await self._generate_recovery_success_message(
                    recovery_method, state, language
                )
                state["messages"].append(AIMessage(content=success_msg))

                # Route to appropriate node based on recovery
                state["next_node"] = recovery_result.get(
                    "next_node", "wait_for_user_input"
                )

                # Update state with recovery results
                if "data" in recovery_result:
                    state.update(recovery_result["data"])

                # Reset error state
                state["last_error"] = None
                state["failed_node"] = None

            else:
                # Recovery failed
                failure_msg = await self._generate_recovery_failure_message(
                    recovery_method, retry_count, state, language
                )
                state["messages"].append(AIMessage(content=failure_msg))

                # Route based on retry count
                if retry_count >= 3:
                    # Give up and provide manual alternatives
                    state["next_node"] = "manual_assistance_node"
                else:
                    # Try error handler again with incremented retry
                    state["next_node"] = "error_handler_node"

            return state

        except Exception as e:
            logger.error(f"Error in recovery node: {str(e)}")
            # Recovery node itself failed - provide simple fallback
            fallback_msg = self._get_recovery_fallback_message(
                state.get("conversation_language", "english")
            )
            state["messages"].append(AIMessage(content=fallback_msg))
            state["next_node"] = "wait_for_user_input"
            state["last_error"] = None

            return state

    async def _execute_recovery(
        self, method: str, state: AgentState, failed_node: str
    ) -> Dict[str, Any]:
        """
        Execute specific recovery method.

        Args:
            method: Recovery method name
            state: Current state
            failed_node: Node that failed

        Returns:
            Recovery result with success status
        """
        logger.info(f"Executing recovery method: {method}")

        # Recovery method implementations
        if method == "retry_with_backoff":
            # Simple retry of the original operation
            return {
                "success": True,
                "next_node": failed_node,  # Retry the same node
            }

        elif method == "search_without_filters":
            # Remove filters and search again
            state["active_filters"] = {}
            return {"success": True, "next_node": "driver_search_node"}

        elif method == "retry_from_cache":
            # Use cached data if available
            if state.get("cache_keys_used"):
                return {"success": True, "next_node": "filter_application_node"}
            return {"success": False}

        elif method == "fresh_search":
            # Force fresh search without cache
            state["use_cache"] = False
            return {"success": True, "next_node": "driver_search_node"}

        elif method == "use_cached_driver":
            # Use driver from current_drivers if available
            if state.get("current_drivers") and state.get("selected_driver"):
                return {"success": True, "next_node": "wait_for_user_input"}
            return {"success": False}

        elif method == "show_contact_only":
            # Show basic contact without full booking
            if state.get("selected_driver"):
                contact_info = {
                    "driver_name": state["selected_driver"].name,
                    "phone": state["selected_driver"].phone_no,
                }
                return {
                    "success": True,
                    "data": {"booking_fallback": contact_info},
                    "next_node": "wait_for_user_input",
                }
            return {"success": False}

        elif method == "manual_booking_instructions":
            # Provide manual booking guidance
            return {
                "success": True,
                "data": {"manual_mode": True},
                "next_node": "manual_assistance_node",
            }

        elif method == "suggest_different_city":
            # Route to nearby cities suggestion
            return {"success": True, "next_node": "suggest_nearby_cities_node"}

        else:
            # Generic recovery or unknown method
            return {"success": False}

    async def _generate_recovery_success_message(
        self, method: str, state: AgentState, language: str
    ) -> str:
        """
        Generate message for successful recovery.

        Args:
            method: Recovery method used
            state: Current state
            language: Response language

        Returns:
            Success message
        """
        context = {
            "search_without_filters": "removing filters",
            "fresh_search": "refreshing data",
            "use_cached_driver": "using saved information",
            "show_contact_only": "getting contact details",
        }

        action_taken = context.get(method, "trying alternative approach")

        prompt = f"""Generate a brief success message after error recovery.

                Context:
                    - Recovery action: {action_taken}
                    - Language: {language}
                    - Don't mention technical details

                Requirements:
                    1. Very brief confirmation
                    2. Focus on moving forward
                    3. Natural, not technical
                    4. Use {language} language

                Generate only the message.
            """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return "ठीक है, आगे बढ़ते हैं।"
            else:
                return "Alright, let's continue."

    async def _generate_recovery_failure_message(
        self, method: str, retry_count: int, state: AgentState, language: str
    ) -> str:
        """
        Generate message for failed recovery.

        Args:
            method: Recovery method attempted
            retry_count: Number of retries
            state: Current state
            language: Response language

        Returns:
            Failure message
        """
        is_final = retry_count >= 3

        prompt = f"""Generate a message after recovery attempt failed.

                Context:
                    - Retry count: {retry_count}
                    - Is final attempt: {is_final}
                    - Language: {language}

                Requirements:
                    1. Brief acknowledgment
                    2. If final, mention we'll try different approach
                    3. Stay positive and helpful
                    4. No technical details
                    5. Use {language} language

                    Generate only the message.
                """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                if is_final:
                    return "तकनीकी समस्या बनी हुई है। आइए दूसरा तरीका आज़माते हैं।"
                else:
                    return "अभी भी समस्या है, एक और कोशिश करते हैं।"
            else:
                if is_final:
                    return "The issue persists. Let me help you in a different way."
                else:
                    return "Still having issues, let me try once more."

    def _get_recovery_fallback_message(self, language: str) -> str:
        """
        Get fallback message when recovery itself fails.

        Args:
            language: Response language

        Returns:
            Fallback message
        """
        if language == "hindi":
            return "तकनीकी समस्या है। कृपया नई खोज शुरू करें या सीधे ड्राइवर से संपर्क करें।"
        else:
            return "I'm experiencing technical difficulties. Please start a new search or contact the driver directly if you have their details."
