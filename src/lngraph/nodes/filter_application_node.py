"""
Filter Application Node for the cab booking agent.
Handles filtering existing driver results from cache.
"""

from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
import logging

from src.models.agent_state_model import AgentState
from src.models.drivers_model import DriverModel
from src.lngraph.tools.driver_tools import DriverTools

logger = logging.getLogger(__name__)


class FilterApplicationNode:
    """
    Node responsible for:
    1. Applying filters to cached driver results
    2. Triggering fresh search if results are too few
    3. Managing filter combinations
    4. Presenting filtered results
    """

    MIN_RESULTS_THRESHOLD = 5  # Minimum drivers before triggering fresh search

    def __init__(self, llm: BaseChatModel, driver_tools: DriverTools):
        """
        Initialize the filter application node.

        Args:
            llm: Language model for generating responses
            driver_tools: Driver tools instance for API calls
        """
        self.llm = llm
        self.driver_tools = driver_tools

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Apply filters to existing driver results.

        Args:
            state: Current agent state

        Returns:
            Updated state with filtered results
        """
        try:
            # Get current parameters
            city = state.get("pickup_city")
            page = state.get("current_page", 1)
            filters = state.get("active_filters", {})

            # Check if we have cached results to filter
            if not state.get("current_drivers"):
                logger.warning("No cached drivers to filter, triggering search")
                state["next_node"] = "driver_search_node"
                return state

            logger.info(f"Applying filters: {filters}")

            # First try filtering from cache
            filter_result = (
                await self.driver_tools.get_drivers_with_user_filter_via_cache_tool(
                    city=city, page=page, filter_obj=filters
                )
            )

            if filter_result["success"]:
                filtered_drivers = filter_result["filtered_drivers"]
                total_filtered = filter_result["total"]

                # Check if we have enough results
                if total_filtered < self.MIN_RESULTS_THRESHOLD and state.get(
                    "has_more_results", False
                ):
                    # Not enough results, trigger fresh API search with filters
                    logger.info(
                        f"Only {total_filtered} drivers found, triggering fresh search"
                    )

                    # Generate message about fetching more results
                    fetching_message = await self._generate_fetching_message(
                        total_filtered,
                        filters,
                        state.get("conversation_language", "english"),
                    )
                    state["messages"].append(AIMessage(content=fetching_message))

                    # Trigger fresh search with filters
                    search_result = await self._search_with_filters(city, page, filters)

                    if search_result["success"]:
                        # Update state with new results
                        state["current_drivers"] = search_result["drivers"]
                        state["total_results"] = search_result["total"]
                        state["has_more_results"] = search_result["has_more"]

                        # Generate response for fresh results
                        response_message = await self._generate_filter_response(
                            search_result["drivers"],
                            search_result["total"],
                            filters,
                            "fresh_search",
                            state.get("conversation_language", "english"),
                        )
                    else:
                        # Search failed, use cached filtered results
                        state["current_drivers"] = filtered_drivers
                        state["total_results"] = total_filtered

                        response_message = await self._generate_filter_response(
                            filtered_drivers,
                            total_filtered,
                            filters,
                            "cache_only",
                            state.get("conversation_language", "english"),
                        )
                else:
                    # We have enough results from cache
                    state["current_drivers"] = filtered_drivers
                    state["total_results"] = total_filtered

                    response_message = await self._generate_filter_response(
                        filtered_drivers,
                        total_filtered,
                        filters,
                        "cache",
                        state.get("conversation_language", "english"),
                    )

                # Add response to messages
                state["messages"].append(AIMessage(content=response_message))

                # Determine next node
                if total_filtered == 0:
                    state["next_node"] = "filter_relaxation_node"
                else:
                    state["next_node"] = "wait_for_user_input"

            else:
                # Filter application failed
                logger.error(f"Filter failed: {filter_result.get('error')}")
                state["last_error"] = filter_result.get(
                    "msg", "Failed to apply filters"
                )

                error_message = await self._generate_error_response(
                    filter_result.get("msg"),
                    state.get("conversation_language", "english"),
                )

                state["messages"].append(AIMessage(content=error_message))
                state["next_node"] = "error_recovery_node"

            return state

        except Exception as e:
            logger.error(f"Error in filter application node: {str(e)}")
            state["last_error"] = f"Filter application failed: {str(e)}"
            state["next_node"] = "error_handler_node"
            return state

    async def _search_with_filters(
        self, city: str, page: int, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform fresh search with filters using search_drivers_tool.

        Args:
            city: City to search in
            page: Page number
            filters: Filters to apply

        Returns:
            Search result dictionary
        """
        # Prepare search parameters
        search_params = {
            "city": city,
            "page": page,
            "limit": 10,
            "use_cache": False,  # Fresh search
        }

        # Map filters to search parameters
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

        return await self.driver_tools.search_drivers_tool(**search_params)

    async def _generate_filter_response(
        self,
        drivers: List[DriverModel],
        total: int,
        filters: Dict[str, Any],
        source: str,
        language: str,
    ) -> str:
        """
        Generate response for filtered results.

        Args:
            drivers: Filtered drivers
            total: Total filtered count
            filters: Applied filters
            source: "cache", "fresh_search", or "cache_only"
            language: Response language

        Returns:
            Response message
        """
        # Create driver summaries
        driver_summaries = []
        for i, driver in enumerate(drivers[:5]):
            vehicles = ", ".join([v.vehicle_type for v in driver.verified_vehicles])

            summary = {
                "index": i + 1,
                "name": driver.name,
                "experience": f"{driver.experience} years",
                "vehicles": vehicles,
                "matches": self._get_matching_filters(driver, filters),
            }
            driver_summaries.append(summary)

        # Format filters for display
        filter_display = []
        for k, v in filters.items():
            if isinstance(v, list):
                filter_display.append(f"{k}: {', '.join(v)}")
            else:
                filter_display.append(f"{k}: {v}")

        prompt = f"""Generate a response for filtered driver results.

                    Context:
                    - Applied filters: {", ".join(filter_display)}
                    - Total matching drivers: {total}
                    - Showing: {len(drivers)} drivers
                    - Source: {source} (cache/fresh_search/cache_only)
                    - Language: {language}

                    Driver summaries:
                        {driver_summaries}

                    Requirements:
                        1. Clearly state the filters applied
                        2. Show how many drivers match the criteria
                        3. Present drivers highlighting how they match the filters
                        4. If source is "fresh_search", mention we fetched more results
                        5. Use {language} language
                        6. Suggest next actions (refine filters, view details, book)
                        7. If few results, suggest relaxing filters

                        Generate only the response message.
                """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            if language == "hindi":
                return f"आपके फिल्टर के अनुसार {total} ड्राइवर मिले हैं।"
            else:
                return f"Found {total} drivers matching your filters: {', '.join(filter_display)}"

    async def _generate_fetching_message(
        self, current_count: int, filters: Dict[str, Any], language: str
    ) -> str:
        """
        Generate message about fetching more results.

        Args:
            current_count: Current filtered count
            filters: Applied filters
            language: Response language

        Returns:
            Fetching message
        """
        prompt = f"""Generate a brief message informing the user we're fetching more results.

                    Context:
                        - Found only {current_count} drivers with current filters
                        - Fetching more results from server
                        - Language: {language}

                    Requirements:
                        1. Brief and informative
                        2. Mention we found few results and are getting more
                        3. Use {language} language
                        4. Keep it conversational

                    Generate only the message.
            """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return f"केवल {current_count} ड्राइवर मिले। अधिक परिणाम खोज रहे हैं..."
            else:
                return f"Found only {current_count} drivers. Fetching more results to give you better options..."

    async def _generate_error_response(self, error_msg: str, language: str) -> str:
        """
        Generate error response for filter failure.

        Args:
            error_msg: Error message
            language: Response language

        Returns:
            User-friendly error message
        """
        prompt = f"""Generate a helpful error message for failed filter application.

                Error: {error_msg}
                Language: {language}

                Requirements:
                    1. Apologize briefly
                    2. Suggest alternative actions (try different filters, search again)
                    3. Use {language} language
                    4. Keep it helpful and conversational

                    Generate only the error message.
            """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return "फिल्टर लागू करने में समस्या हुई। कृपया दूसरे फिल्टर आज़माएं।"
            else:
                return "I had trouble applying those filters. Please try different filter options or let me search again."

    def _get_matching_filters(
        self, driver: DriverModel, filters: Dict[str, Any]
    ) -> List[str]:
        """
        Get list of filters that this driver matches.

        Args:
            driver: Driver model
            filters: Applied filters

        Returns:
            List of matching filter descriptions
        """
        matches = []

        if "vehicle_types" in filters:
            driver_vehicles = [v.vehicle_type for v in driver.verified_vehicles]
            matching_vehicles = [
                v for v in filters["vehicle_types"] if v in driver_vehicles
            ]
            if matching_vehicles:
                matches.append(f"has {', '.join(matching_vehicles)}")

        if "gender" in filters and driver.gender == filters["gender"]:
            matches.append(f"{filters['gender']} driver")

        if (
            "is_pet_allowed" in filters
            and driver.is_pet_allowed == filters["is_pet_allowed"]
        ):
            matches.append("allows pets" if filters["is_pet_allowed"] else "no pets")

        if (
            "min_experience" in filters
            and driver.experience >= filters["min_experience"]
        ):
            matches.append(f"{driver.experience}+ years experience")

        return matches
