"""
No Results Handler Node for the cab booking agent.
Handles cases when no drivers are found in the search.
"""

from typing import Dict, Any, List, Optional
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
import logging

from src.models.agent_state_model import AgentState
from src.lngraph.tools.driver_tools import DriverTools

logger = logging.getLogger(__name__)


class NoResultsHandlerNode:
    """
    Node responsible for:
    1. Handling zero search results gracefully
    2. Suggesting nearby cities
    3. Recommending filter removal if any applied
    4. Offering alternative search strategies
    """

    # Major cities and their nearby alternatives
    NEARBY_CITIES = {
        "delhi": ["gurgaon", "noida", "faridabad", "ghaziabad"],
        "mumbai": ["thane", "navi mumbai", "kalyan", "panvel"],
        "bangalore": ["mysore", "tumkur", "hosur"],
        "bengaluru": ["mysore", "tumkur", "hosur"],
        "kolkata": ["howrah", "durgapur", "asansol"],
        "chennai": ["kanchipuram", "vellore", "pondicherry"],
        "hyderabad": ["secunderabad", "warangal", "nizamabad"],
        "pune": ["pimpri-chinchwad", "nashik", "satara"],
        "ahmedabad": ["gandhinagar", "vadodara", "rajkot"],
        "gurgaon": ["delhi", "faridabad", "rewari"],
        "gurugram": ["delhi", "faridabad", "rewari"],
        "noida": ["delhi", "ghaziabad", "greater noida"],
    }

    def __init__(self, llm: BaseChatModel, driver_tools: DriverTools):
        """
        Initialize the no results handler node.

        Args:
            llm: Language model for generating responses
            driver_tools: Driver tools instance for checking nearby cities
        """
        self.llm = llm
        self.driver_tools = driver_tools

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Handle no search results scenario.

        Args:
            state: Current agent state

        Returns:
            Updated state with suggestions
        """
        try:
            city = state.get("pickup_city")
            filters = state.get("active_filters", {})
            language = state.get("conversation_language", "english")
            search_attempt = state.get("retry_count", 0)

            logger.info(f"No results found for {city} with filters: {filters}")

            # Determine the best strategy based on context
            suggestions = []

            # Strategy 1: If filters are applied, suggest removing them
            if filters:
                suggestions.append(
                    {
                        "type": "remove_filters",
                        "action": "search_without_filters",
                        "description": "Search without filters to see all available drivers",
                    }
                )

            # Strategy 2: Check nearby cities
            nearby_cities = await self._find_nearby_cities_with_drivers(city)
            if nearby_cities:
                suggestions.append(
                    {
                        "type": "nearby_cities",
                        "cities": nearby_cities[:3],  # Top 3 cities
                        "description": "Search in nearby cities",
                    }
                )

            # Strategy 3: Suggest expanding search radius (for geo search)
            if state.get("search_strategy") == "geo":
                suggestions.append(
                    {
                        "type": "expand_radius",
                        "current_radius": state.get("radius", 100),
                        "suggested_radius": state.get("radius", 100) + 50,
                        "description": "Expand search area",
                    }
                )

            # Strategy 4: If this is a retry, suggest different approach
            if search_attempt > 0:
                suggestions.append(
                    {
                        "type": "contact_support",
                        "description": "Try again later or contact support",
                    }
                )

            # Generate response with suggestions
            response = await self._generate_no_results_response(
                city, filters, suggestions, language
            )

            state["messages"].append(AIMessage(content=response))

            # Store suggestions in state for follow-up
            state["no_results_suggestions"] = suggestions

            # Route to appropriate next node
            if nearby_cities:
                state["suggested_cities"] = nearby_cities
                state["next_node"] = "wait_for_user_input"
            else:
                state["next_node"] = "wait_for_user_input"

            return state

        except Exception as e:
            logger.error(f"Error in no results handler: {str(e)}")
            state["last_error"] = f"Failed to handle no results: {str(e)}"
            state["next_node"] = "error_handler_node"
            return state

    async def _find_nearby_cities_with_drivers(self, city: str) -> List[Dict[str, Any]]:
        """
        Find nearby cities that have drivers available.

        Args:
            city: Current city with no results

        Returns:
            List of nearby cities with driver counts
        """
        nearby = []

        # Get predefined nearby cities
        nearby_city_names = self.NEARBY_CITIES.get(city.lower(), [])

        # If no predefined nearby cities, ask LLM
        if not nearby_city_names:
            nearby_city_names = await self._get_nearby_cities_from_llm(city)

        # Check each nearby city for drivers
        for nearby_city in nearby_city_names[:5]:  # Check max 5 cities
            try:
                result = await self.driver_tools.search_drivers_tool(
                    city=nearby_city,
                    page=1,
                    limit=1,  # Just need count
                    use_cache=True,
                )

                if result["success"] and result["total"] > 0:
                    nearby.append(
                        {
                            "city": nearby_city,
                            "driver_count": result["total"],
                            "distance_description": self._estimate_distance(
                                city, nearby_city
                            ),
                        }
                    )

            except Exception as e:
                logger.warning(f"Error checking {nearby_city}: {e}")
                continue

        # Sort by driver count (descending)
        nearby.sort(key=lambda x: x["driver_count"], reverse=True)

        return nearby

    async def _get_nearby_cities_from_llm(self, city: str) -> List[str]:
        """
        Get nearby cities using LLM when not in predefined list.

        Args:
            city: City name

        Returns:
            List of nearby city names
        """
        prompt = f"""List 3-5 major cities near {city} in India.

            Requirements:
                1. Only cities within 200km
                2. Preferably in the same state
                3. Major cities with good connectivity
                4. Return as comma-separated list

            Return only the city names, nothing else.
        """

        try:
            response = await self.llm.ainvoke(prompt)
            cities = [c.strip().lower() for c in response.content.split(",")]
            return cities[:5]
        except:
            return []

    def _estimate_distance(self, city1: str, city2: str) -> str:
        """
        Estimate distance description between cities.

        Args:
            city1: First city
            city2: Second city

        Returns:
            Distance description
        """
        # This is a simplified estimation
        known_distances = {
            ("delhi", "gurgaon"): "30 km",
            ("delhi", "noida"): "25 km",
            ("delhi", "faridabad"): "35 km",
            ("mumbai", "thane"): "25 km",
            ("mumbai", "navi mumbai"): "30 km",
            ("bangalore", "mysore"): "150 km",
        }

        key = (city1.lower(), city2.lower())
        reverse_key = (city2.lower(), city1.lower())

        if key in known_distances:
            return known_distances[key]
        elif reverse_key in known_distances:
            return known_distances[reverse_key]
        else:
            return "nearby"

    async def _generate_no_results_response(
        self,
        city: str,
        filters: Dict[str, Any],
        suggestions: List[Dict[str, Any]],
        language: str,
    ) -> str:
        """
        Generate response for no results scenario.

        Args:
            city: Search city
            filters: Applied filters
            suggestions: List of suggestions
            language: Response language

        Returns:
            Response message
        """
        # Format filters for display
        filter_str = ""
        if filters:
            filter_parts = []
            for k, v in filters.items():
                if isinstance(v, list):
                    filter_parts.append(f"{k}: {', '.join(v)}")
                else:
                    filter_parts.append(f"{k}: {v}")
            filter_str = f" with filters: {', '.join(filter_parts)}"

        prompt = f"""Generate a helpful response for no search results.

                Context:
                    - Searched in: {city}{filter_str}
                    - Language: {language}

                Available suggestions:
                    {suggestions}

                Requirements:
                    1. Acknowledge no drivers found (don't apologize excessively)
                    2. Present suggestions clearly and actionably
                    3. If nearby cities available, list them with driver counts
                    4. Prioritize most helpful suggestions
                    5. Keep positive and solution-focused
                    6. Use {language} language
                    7. Make it easy for user to take next action

                    Generate only the response message.
                """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            # Fallback
            if language == "hindi":
                fallback = f"{city} में कोई ड्राइवर नहीं मिला।"
                if suggestions:
                    fallback += " आप आस-पास के शहरों में खोज सकते हैं।"
                return fallback
            else:
                fallback = f"No drivers found in {city}."
                if suggestions:
                    fallback += " Here are some alternatives:\n"
                    for s in suggestions:
                        if s["type"] == "nearby_cities":
                            for c in s["cities"]:
                                fallback += (
                                    f"• {c['city']}: {c['driver_count']} drivers\n"
                                )
                return fallback
