"""
Pagination Node for the cab booking agent.
Handles requests for more drivers or next page of results.
"""

from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
import logging

from src.models.agent_state_model import AgentState
from src.models.drivers_model import DriverModel
from src.lngraph.tools.driver_tools import DriverTools

logger = logging.getLogger(__name__)


class PaginationNode:
    """
    Node responsible for:
    1. Fetching next page of results
    2. Applying existing filters to new page
    3. Managing page state
    4. Handling end of results
    """

    def __init__(self, llm: BaseChatModel, driver_tools: DriverTools):
        """
        Initialize the pagination node.

        Args:
            llm: Language model for generating responses
            driver_tools: Driver tools instance for API calls
        """
        self.llm = llm
        self.driver_tools = driver_tools

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Handle pagination request for more drivers.

        Args:
            state: Current agent state

        Returns:
            Updated state with next page results
        """
        try:
            # Check if more results available
            if not state.get("has_more_results", False):
                # No more results
                no_more_msg = await self._generate_no_more_results_message(
                    state.get("total_results", 0),
                    state.get("conversation_language", "english"),
                )
                state["messages"].append(AIMessage(content=no_more_msg))
                state["next_node"] = "wait_for_user_input"
                return state

            # Get current parameters
            city = state.get("pickup_city")
            current_page = state.get("current_page", 1)
            next_page = current_page + 1
            page_size = state.get("page_size", 10)
            filters = state.get("active_filters", {})

            logger.info(f"Fetching page {next_page} for {city}")

            # Check if we should use filtered search or regular search
            if filters:
                # Use search with filters for consistency
                search_params = {
                    "city": city,
                    "page": next_page,
                    "limit": page_size,
                }

                # Apply filters
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

                for filter_key, api_key in filter_mapping.items():
                    if filter_key in filters:
                        search_params[api_key] = filters[filter_key]

                result = await self.driver_tools.search_drivers_tool(**search_params)
            else:
                # Regular search without filters
                result = await self.driver_tools.search_drivers_tool(
                    city=city, page=next_page, limit=page_size
                )

            if result["success"]:
                drivers = result["drivers"]

                if len(drivers) == 0:
                    # No drivers on this page
                    no_results_msg = await self._generate_empty_page_message(
                        current_page, state.get("conversation_language", "english")
                    )
                    state["messages"].append(AIMessage(content=no_results_msg))
                    state["next_node"] = "wait_for_user_input"
                    return state

                # Update state with new results
                state["current_drivers"] = drivers
                state["current_page"] = next_page
                state["has_more_results"] = result["has_more"]

                # Update cache keys
                if "cache_keys_used" not in state:
                    state["cache_keys_used"] = []
                cache_key = f"{state['session_id']}_{city}_{next_page}"
                if cache_key not in state["cache_keys_used"]:
                    state["cache_keys_used"].append(cache_key)

                # Calculate drivers shown so far
                total_shown = (next_page - 1) * page_size + len(drivers)

                # Generate response
                response_msg = await self._generate_pagination_response(
                    drivers,
                    next_page,
                    total_shown,
                    result["total"],
                    filters,
                    state.get("conversation_language", "english"),
                )

                state["messages"].append(AIMessage(content=response_msg))
                state["next_node"] = "wait_for_user_input"

            else:
                # Pagination failed
                logger.error(f"Pagination failed: {result.get('error')}")
                error_msg = await self._generate_pagination_error_message(
                    state.get("conversation_language", "english")
                )
                state["messages"].append(AIMessage(content=error_msg))
                state["next_node"] = "wait_for_user_input"

            return state

        except Exception as e:
            logger.error(f"Error in pagination node: {str(e)}")
            state["last_error"] = f"Failed to load more results: {str(e)}"
            state["next_node"] = "error_handler_node"
            return state

    async def _generate_pagination_response(
        self,
        drivers: List[DriverModel],
        page: int,
        total_shown: int,
        total_results: int,
        filters: Dict[str, Any],
        language: str,
    ) -> str:
        """
        Generate response for paginated results.

        Args:
            drivers: List of drivers on this page
            page: Current page number
            total_shown: Total drivers shown so far
            total_results: Total available results
            filters: Active filters
            language: Response language

        Returns:
            Response message
        """
        # Create driver summaries
        driver_summaries = []
        for i, driver in enumerate(drivers[:5]):  # Show top 5
            vehicles = ", ".join([v.vehicle_type for v in driver.verified_vehicles])

            # Calculate display number (continuing from previous pages)
            display_num = (page - 1) * 10 + i + 1

            summary = {
                "number": display_num,
                "name": driver.name,
                "experience": f"{driver.experience} years",
                "vehicles": vehicles,
                "verified": "✓" if driver.profile_verified else "✗",
            }
            driver_summaries.append(summary)

        # Format active filters
        filter_str = ""
        if filters:
            filter_parts = []
            for k, v in filters.items():
                if isinstance(v, list):
                    filter_parts.append(f"{k}: {', '.join(v)}")
                else:
                    filter_parts.append(f"{k}: {v}")
            filter_str = f"Active filters: {', '.join(filter_parts)}"

        prompt = f"""Generate a response showing next page of driver results.

                    Context:
                        - Page {page} of results
                        - Showing drivers {total_shown - len(drivers) + 1} to {total_shown} of {total_results} total
                        - {filter_str}
                        - Language: {language}

                    Drivers on this page:
                        {driver_summaries}

                    Requirements:
                        1. Clearly indicate this is page {page}
                        2. Show drivers with their numbers continuing from previous pages
                        3. Mention how many total results and how many shown
                        4. If filters active, remind about them briefly
                        5. Present drivers clearly
                        6. Mention if more pages available
                        7. Use {language} language
                        8. Keep it concise but informative

                    Generate only the response message.
                """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            # Fallback
            if language == "hindi":
                return f"पेज {page}: {len(drivers)} और ड्राइवर मिले। कुल {total_results} में से {total_shown} दिखाए गए।"
            else:
                return f"Page {page}: Here are {len(drivers)} more drivers. Showing {total_shown} of {total_results} total results."

    async def _generate_no_more_results_message(
        self, total_shown: int, language: str
    ) -> str:
        """
        Generate message when no more results available.

        Args:
            total_shown: Total drivers shown
            language: Response language

        Returns:
            No more results message
        """
        prompt = f"""Generate a message informing user there are no more results.

                Context:
                    - Total drivers shown: {total_shown}
                    - Language: {language}

                Requirements:
                    1. Politely inform no more drivers available
                    2. Mention total shown
                    3. Suggest alternatives (refine search, change filters, book from shown)
                    4. Keep it helpful and positive
                    5. Use {language} language

                    Generate only the message.

            """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return f"आपने सभी {total_shown} उपलब्ध ड्राइवर देख लिए हैं। आप इनमें से चुन सकते हैं या अपनी खोज बदल सकते हैं।"
            else:
                return f"You've seen all {total_shown} available drivers. You can choose from these or modify your search criteria for different results."

    async def _generate_empty_page_message(
        self, current_page: int, language: str
    ) -> str:
        """
        Generate message when a page has no results.

        Args:
            current_page: Current page number
            language: Response language

        Returns:
            Empty page message
        """
        prompt = f"""Generate a message for when a page has no results.

                Context:
                    - Was on page {current_page}
                    - Next page is empty
                    - Language: {language}

                Requirements:
                    1. Inform no more drivers on next page
                    2. Suggest going back to previous results
                    3. Keep it brief and helpful
                    4. Use {language} language

                Generate only the message."""

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return "अगले पेज पर कोई ड्राइवर नहीं है। पिछले परिणामों से चुनें।"
            else:
                return "No more drivers found on the next page. Please choose from the previous results or modify your search."

    async def _generate_pagination_error_message(self, language: str) -> str:
        """
        Generate error message for pagination failure.

        Args:
            language: Response language

        Returns:
            Error message
        """
        prompt = f"""Generate an error message for failed pagination.

                Language: {language}

                Requirements:
                    1. Apologize for the issue
                    2. Suggest trying again or working with current results
                    3. Keep it positive
                    4. Use {language} language

                Generate only the message.
            """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return "अधिक परिणाम लोड करने में समस्या हुई। कृपया दोबारा प्रयास करें या मौजूदा विकल्पों से चुनें।"
            else:
                return "I had trouble loading more results. Please try again or choose from the current options."
