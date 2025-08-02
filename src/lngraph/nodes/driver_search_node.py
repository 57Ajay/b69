"""
Driver Search Node for the cab booking agent.
Handles searching for drivers using the search_drivers_tool.
"""

from typing import Dict, Any, Optional, List
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
import logging

from src.models.agent_state_model import AgentState
from src.models.drivers_model import DriverModel
from src.lngraph.tools.driver_tools import DriverTools

logger = logging.getLogger(__name__)


class DriverSearchNode:
    """
    Node responsible for:
    1. Searching for drivers using API
    2. Applying initial filters from user request
    3. Handling search errors
    4. Formatting and presenting results
    """

    def __init__(self, llm: BaseChatModel, driver_tools: DriverTools):
        """
        Initialize the driver search node.

        Args:
            llm: Language model for generating responses
            driver_tools: Driver tools instance for API calls
        """
        self.llm = llm
        self.driver_tools = driver_tools

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Search for drivers based on state criteria.

        Args:
            state: Current agent state

        Returns:
            Updated state with search results
        """
        try:
            # Get search parameters from state
            city = state.get("pickup_city")
            page = state.get("current_page", 1)
            page_size = state.get("page_size", 10)
            filters = state.get("active_filters", {})

            # Log search attempt
            logger.info(f"Searching drivers in {city}, page {page}, filters: {filters}")

            # Prepare search parameters
            search_params = {
                "city": city,
                "page": page,
                "limit": page_size,
            }

            # Map filter keys to API parameters
            filter_mapping = {
                "vehicle_types": "vehicle_types",
                "gender": "gender",
                "min_age": "min_age",
                "max_age": "max_age",
                "is_pet_allowed": "is_pet_allowed",
                "min_connections": "min_connections",
                "min_experience": "min_experience",
                "languages": "languages",
                "profile_verified": "profile_verified",
                "married": "married",
            }

            # Apply filters to search parameters
            for filter_key, api_key in filter_mapping.items():
                if filter_key in filters:
                    search_params[api_key] = filters[filter_key]

            # Call search tool
            result = await self.driver_tools.search_drivers_tool(**search_params)

            if result["success"]:
                # Update state with results
                drivers = result["drivers"]
                state["current_drivers"] = drivers
                state["total_results"] = result["total"]
                state["has_more_results"] = result["has_more"]
                state["current_page"] = page

                # Update cache keys
                if "cache_keys_used" not in state:
                    state["cache_keys_used"] = []
                cache_key = f"{state['session_id']}_{city}_{page}"
                if cache_key not in state["cache_keys_used"]:
                    state["cache_keys_used"].append(cache_key)

                # Generate response message
                response_message = await self._generate_search_response(
                    drivers,
                    result["total"],
                    filters,
                    state.get("conversation_language", "english"),
                )

                # Add response to messages
                state["messages"].append(AIMessage(content=response_message))

                # Determine next node
                if len(drivers) == 0:
                    state["next_node"] = "no_results_handler_node"
                else:
                    state["next_node"] = "wait_for_user_input"

            else:
                # Handle search failure
                logger.error(f"Search failed: {result.get('error')}")
                state["last_error"] = result.get("msg", "Failed to search drivers")

                # Generate error message
                error_message = await self._generate_error_response(
                    result.get("msg"), state.get("conversation_language", "english")
                )

                state["messages"].append(AIMessage(content=error_message))
                state["next_node"] = "error_recovery_node"

            return state

        except Exception as e:
            logger.error(f"Error in driver search node: {str(e)}")
            state["last_error"] = f"Search failed: {str(e)}"
            state["next_node"] = "error_handler_node"
            return state

    async def _generate_search_response(
        self,
        drivers: List[DriverModel],
        total: int,
        filters: Dict[str, Any],
        language: str,
    ) -> str:
        """
        Generate natural language response for search results.

        Args:
            drivers: List of driver models
            total: Total number of results
            filters: Applied filters
            language: Response language

        Returns:
            Formatted response message
        """
        # Create driver summaries
        driver_summaries = []
        for i, driver in enumerate(drivers[:5]):  # Show top 5
            vehicles = ", ".join([v.vehicle_type for v in driver.verified_vehicles])
            languages = ", ".join(driver.verified_languages)

            summary = {
                "index": i + 1,
                "name": driver.name,
                "age": driver.age if driver.age > 0 else "Not specified",
                "experience": driver.experience,
                "vehicles": vehicles,
                "languages": languages,
                "pet_allowed": "Yes" if driver.is_pet_allowed else "No",
                "verified": "✓" if driver.profile_verified else "✗",
            }
            driver_summaries.append(summary)

        # Create context for response generation
        filters_str = ""
        if filters:
            filter_parts = []
            for k, v in filters.items():
                if isinstance(v, list):
                    filter_parts.append(f"{k}: {', '.join(v)}")
                else:
                    filter_parts.append(f"{k}: {v}")
            filters_str = f"Applied filters: {', '.join(filter_parts)}"

        prompt = f"""Generate a natural response presenting driver search results.

Results:
- Total drivers found: {total}
- Showing: {len(drivers)} drivers
- {filters_str}

Driver details:
{driver_summaries}

Requirements:
1. Present the drivers in a clear, scannable format
2. Highlight key information (name, experience, vehicles)
3. Mention total results and any applied filters
4. Use {language} language
5. Keep it conversational but informative
6. If more drivers available, mention it
7. Suggest next actions (view details, apply filters, book)

Generate only the response message."""

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            # Fallback response
            if language == "hindi":
                return f"मुझे {total} ड्राइवर मिले हैं। कृपया अपना ड्राइवर चुनें या अधिक विकल्प देखें।"
            else:
                return f"I found {total} drivers. Here are the top matches. You can ask for more details about any driver or apply additional filters."

    async def _generate_error_response(self, error_msg: str, language: str) -> str:
        """
        Generate error response message.

        Args:
            error_msg: Error message
            language: Response language

        Returns:
            User-friendly error message
        """
        prompt = f"""Generate a helpful error message for a failed driver search.

Error: {error_msg}
Language: {language}

Requirements:
1. Apologize for the inconvenience
2. Explain briefly what happened
3. Suggest trying again or alternative actions
4. Keep it conversational and helpful
5. Use {language} language

Generate only the error message."""

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception:
            # Fallback
            if language == "hindi":
                return "क्षमा करें, ड्राइवर खोजने में समस्या हुई। कृपया दोबारा प्रयास करें।"
            else:
                return "I'm sorry, I encountered an issue while searching for drivers. Please try again in a moment."
