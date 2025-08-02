"""
Filter Relaxation Node for the cab booking agent.
Handles cases when filters are too restrictive and suggests alternatives.
"""

from typing import Dict, Any, List, Tuple
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
import logging

from src.models.agent_state_model import AgentState
from src.models.drivers_model import DriverModel
from src.lngraph.tools.driver_tools import DriverTools

logger = logging.getLogger(__name__)


class FilterRelaxationNode:
    """
    Node responsible for:
    1. Analyzing why no results were found
    2. Suggesting filter relaxations
    3. Showing what's available without filters
    4. Offering alternative search strategies
    """

    # Filter priority for relaxation (least important to most important)
    FILTER_RELAXATION_PRIORITY = [
        "married",
        "profile_verified",
        "languages",
        "max_age",
        "min_age",
        "min_connections",
        "gender",
        "min_experience",
        "is_pet_allowed",
        "vehicle_types",
    ]

    def __init__(self, llm: BaseChatModel, driver_tools: DriverTools):
        """
        Initialize the filter relaxation node.

        Args:
            llm: Language model for generating responses
            driver_tools: Driver tools instance for API calls
        """
        self.llm = llm
        self.driver_tools = driver_tools

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Handle filter relaxation when no results found.

        Args:
            state: Current agent state

        Returns:
            Updated state with relaxation suggestions
        """
        try:
            city = state.get("pickup_city")
            current_filters = state.get("active_filters", {})
            language = state.get("conversation_language", "english")

            logger.info(f"No results with filters: {current_filters}")

            # First, check what's available without any filters
            total_available = await self._check_total_available(city)

            if total_available == 0:
                # No drivers at all in this city
                response_message = await self._generate_no_drivers_message(
                    city, language
                )
                state["messages"].append(AIMessage(content=response_message))
                state["next_node"] = "suggest_nearby_cities_node"
                return state

            # Try relaxing filters one by one
            relaxation_suggestions = await self._find_relaxation_options(
                city, current_filters, total_available
            )

            # Generate response with suggestions
            response_message = await self._generate_relaxation_response(
                current_filters, relaxation_suggestions, total_available, language
            )

            state["messages"].append(AIMessage(content=response_message))

            # Store relaxation suggestions in state
            state["filter_relaxation_suggestions"] = relaxation_suggestions

            # Set next node to wait for user choice
            state["next_node"] = "wait_for_user_input"

            return state

        except Exception as e:
            logger.error(f"Error in filter relaxation node: {str(e)}")
            state["last_error"] = f"Failed to process filter relaxation: {str(e)}"
            state["next_node"] = "error_handler_node"
            return state

    async def _check_total_available(self, city: str) -> int:
        """
        Check total drivers available without filters.

        Args:
            city: City to check

        Returns:
            Total number of drivers
        """
        try:
            result = await self.driver_tools.search_drivers_tool(
                city=city,
                page=1,
                limit=1,  # Just need the count
                use_cache=True,
            )

            if result["success"]:
                return result["total"]
            return 0

        except Exception as e:
            logger.error(f"Error checking total drivers: {e}")
            return 0

    async def _find_relaxation_options(
        self, city: str, current_filters: Dict[str, Any], total_available: int
    ) -> List[Dict[str, Any]]:
        """
        Find which filters to relax for better results.

        Args:
            city: City to search
            current_filters: Currently applied filters
            total_available: Total drivers without filters

        Returns:
            List of relaxation suggestions with counts
        """
        suggestions = []

        # Try removing filters by priority
        for filter_to_remove in self.FILTER_RELAXATION_PRIORITY:
            if filter_to_remove not in current_filters:
                continue

            # Create filters without this one
            relaxed_filters = {
                k: v for k, v in current_filters.items() if k != filter_to_remove
            }

            # Skip if we've already tried with no filters
            if not relaxed_filters:
                suggestions.append(
                    {
                        "remove_filter": "all",
                        "filter_value": "all filters",
                        "expected_results": total_available,
                        "description": "Remove all filters",
                    }
                )
                break

            # Check how many results with relaxed filters
            try:
                result = (
                    await self.driver_tools.get_drivers_with_user_filter_via_cache_tool(
                        city=city, page=1, filter_obj=relaxed_filters
                    )
                )

                if result["success"] and result["total"] > 0:
                    suggestions.append(
                        {
                            "remove_filter": filter_to_remove,
                            "filter_value": current_filters[filter_to_remove],
                            "expected_results": result["total"],
                            "description": self._get_filter_description(
                                filter_to_remove, current_filters[filter_to_remove]
                            ),
                        }
                    )

                    # If we found a good number of results, we can stop
                    if result["total"] >= 10:
                        break

            except Exception as e:
                logger.warning(f"Error checking relaxed filter {filter_to_remove}: {e}")

        # Sort suggestions by expected results (descending)
        suggestions.sort(key=lambda x: x["expected_results"], reverse=True)

        return suggestions[:3]  # Return top 3 suggestions

    async def _generate_relaxation_response(
        self,
        current_filters: Dict[str, Any],
        suggestions: List[Dict[str, Any]],
        total_available: int,
        language: str,
    ) -> str:
        """
        Generate response with filter relaxation suggestions.

        Args:
            current_filters: Currently applied filters
            suggestions: Relaxation suggestions
            total_available: Total drivers without filters
            language: Response language

        Returns:
            Response message
        """
        # Format current filters
        filter_list = []
        for k, v in current_filters.items():
            if isinstance(v, list):
                filter_list.append(f"{k}: {', '.join(v)}")
            else:
                filter_list.append(f"{k}: {v}")

        prompt = f"""Generate a helpful response for no search results with filter suggestions.

                    Context:
                        - Applied filters: {", ".join(filter_list)}
                        - Total drivers available without filters: {total_available}
                        - Language: {language}

                    Relaxation suggestions:
                        {suggestions}

                    Requirements:
                        1. Acknowledge no results with current filters
                        2. Suggest the best relaxation options naturally
                        3. Mention how many drivers would be available with each suggestion
                        4. Encourage user to try relaxing filters
                        5. Keep it conversational and helpful
                        6. Use {language} language
                        7. Don't make it too long or overwhelming

                        Generate only the response message.
                """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            # Fallback
            if language == "hindi":
                return f"इन फिल्टर के साथ कोई ड्राइवर नहीं मिला। कुल\
                        {total_available} ड्राइवर उपलब्ध हैं। कृपया कुछ फिल्टर हटाने का प्रयास करें।"
            else:
                return f"No drivers found with these filters. There are\
                        {total_available} drivers available in total. Try removing some filters for better results."

    async def _generate_no_drivers_message(self, city: str, language: str) -> str:
        """
        Generate message when no drivers available at all.

        Args:
            city: City with no drivers
            language: Response language

        Returns:
            No drivers message
        """
        prompt = f"""Generate a message informing no drivers are available in the city.

                    Context:
                        - City: {city}
                        - No drivers available at all
                        - Language: {language}

                        Requirements:
                            1. Politely inform no drivers in this city
                            2. Suggest checking nearby cities
                            3. Apologize for inconvenience
                            4. Use {language} language
                            5. Keep it brief and helpful

                            Generate only the message.
                """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return f"क्षमा करें, {city} में कोई ड्राइवर उपलब्ध नहीं है। कृपया आस-पास के शहरों में खोजें।"
            else:
                return f"I'm sorry, there are no drivers available in {city}. Would you like to search in nearby cities?"

    def _get_filter_description(self, filter_key: str, filter_value: Any) -> str:
        """
        Get human-readable description of a filter.

        Args:
            filter_key: Filter key
            filter_value: Filter value

        Returns:
            Description string
        """
        descriptions = {
            "vehicle_types": f"Vehicle type requirement ({filter_value})",
            "gender": f"Gender preference ({filter_value})",
            "min_age": f"Minimum age ({filter_value})",
            "max_age": f"Maximum age ({filter_value})",
            "is_pet_allowed": "Pet-friendly requirement",
            "min_connections": f"Minimum rides ({filter_value})",
            "min_experience": f"Minimum experience ({filter_value} years)",
            "languages": f"Language requirement ({filter_value})",
            "profile_verified": "Verified profile requirement",
            "married": f"Marital status ({filter_value})",
        }

        return descriptions.get(filter_key, f"{filter_key} filter")
