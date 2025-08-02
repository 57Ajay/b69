"""
Suggest Nearby Cities Node for the cab booking agent.
Provides alternative city suggestions when no drivers found.
"""

from typing import Dict, Any, List, Optional, Tuple
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
import logging

from src.models.agent_state_model import AgentState
from src.lngraph.tools.driver_tools import DriverTools

logger = logging.getLogger(__name__)


class SuggestNearbyCitiesNode:
    """
    Node responsible for:
    1. Finding and suggesting nearby cities with drivers
    2. Providing distance and driver availability info
    3. Handling user selection of alternative cities
    4. Expanding search radius intelligently
    """

    # Major city clusters in India
    CITY_CLUSTERS = {
        "ncr": {
            "cities": [
                "delhi",
                "gurgaon",
                "gurugram",
                "noida",
                "faridabad",
                "ghaziabad",
                "greater noida",
            ],
            "name": "National Capital Region",
        },
        "mumbai_metro": {
            "cities": [
                "mumbai",
                "thane",
                "navi mumbai",
                "kalyan",
                "panvel",
                "vasai",
                "virar",
            ],
            "name": "Mumbai Metropolitan Region",
        },
        "bangalore_region": {
            "cities": [
                "bangalore",
                "bengaluru",
                "mysore",
                "mysuru",
                "tumkur",
                "hosur",
                "mandya",
            ],
            "name": "Bangalore Region",
        },
        "chennai_region": {
            "cities": [
                "chennai",
                "kanchipuram",
                "tiruvallur",
                "chengalpattu",
                "vellore",
            ],
            "name": "Chennai Region",
        },
        "kolkata_region": {
            "cities": ["kolkata", "howrah", "salt lake", "barrackpore", "barasat"],
            "name": "Kolkata Metropolitan Area",
        },
        "pune_region": {
            "cities": [
                "pune",
                "pimpri-chinchwad",
                "pcmc",
                "talegaon",
                "chakan",
                "lonavala",
            ],
            "name": "Pune Region",
        },
        "hyderabad_region": {
            "cities": [
                "hyderabad",
                "secunderabad",
                "cyberabad",
                "shamshabad",
                "medchal",
            ],
            "name": "Hyderabad Region",
        },
    }

    def __init__(self, llm: BaseChatModel, driver_tools: DriverTools):
        """
        Initialize the suggest nearby cities node.

        Args:
            llm: Language model for generating responses
            driver_tools: Driver tools instance for checking availability
        """
        self.llm = llm
        self.driver_tools = driver_tools

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Suggest nearby cities with available drivers.

        Args:
            state: Current agent state

        Returns:
            Updated state with city suggestions
        """
        try:
            current_city = state.get("pickup_city", "").lower()
            language = state.get("conversation_language", "english")
            filters = state.get("active_filters", {})

            # Check if we already have suggested cities from no results handler
            existing_suggestions = state.get("suggested_cities", [])

            if not existing_suggestions:
                # Find nearby cities
                logger.info(f"Finding nearby cities for {current_city}")
                nearby_cities = await self._find_comprehensive_nearby_cities(
                    current_city, filters
                )
            else:
                # Use existing suggestions
                nearby_cities = existing_suggestions

            if not nearby_cities:
                # No nearby cities with drivers
                no_alternatives_msg = await self._generate_no_alternatives_message(
                    current_city, language
                )
                state["messages"].append(AIMessage(content=no_alternatives_msg))
                state["next_node"] = "wait_for_user_input"
                return state

            # Group cities by distance/region for better presentation
            grouped_cities = self._group_cities_by_proximity(
                current_city, nearby_cities
            )

            # Generate suggestion message
            suggestion_msg = await self._generate_city_suggestions_message(
                current_city, grouped_cities, filters, language
            )

            state["messages"].append(AIMessage(content=suggestion_msg))

            # Store suggestions for easy reference
            state["nearby_city_suggestions"] = grouped_cities
            state["suggestion_context"] = "nearby_cities"

            # Add quick selection options
            if len(nearby_cities) <= 5:
                quick_options = [
                    {
                        "number": i + 1,
                        "city": city["city"],
                        "drivers": city["driver_count"],
                    }
                    for i, city in enumerate(nearby_cities)
                ]
                state["quick_city_options"] = quick_options

            state["next_node"] = "wait_for_user_input"

            return state

        except Exception as e:
            logger.error(f"Error in suggest nearby cities: {str(e)}")
            state["last_error"] = f"Failed to find nearby cities: {str(e)}"
            state["next_node"] = "error_handler_node"
            return state

    async def _find_comprehensive_nearby_cities(
        self, current_city: str, filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Find nearby cities with comprehensive search strategies.

        Args:
            current_city: Current city
            filters: Active filters

        Returns:
            List of nearby cities with driver info
        """
        nearby = []
        checked_cities = set()

        # Strategy 1: Check city cluster
        cluster_cities = self._get_cluster_cities(current_city)
        for city in cluster_cities:
            if city not in checked_cities and city != current_city:
                result = await self._check_city_availability(city, filters)
                if result:
                    nearby.append(result)
                checked_cities.add(city)

        # Strategy 2: Get nearby cities from LLM
        if len(nearby) < 3:
            llm_cities = await self._get_nearby_cities_from_llm(current_city)
            for city in llm_cities:
                if city not in checked_cities:
                    result = await self._check_city_availability(city, filters)
                    if result:
                        nearby.append(result)
                    checked_cities.add(city)

        # Strategy 3: Check major cities if still need options
        if len(nearby) < 2:
            major_cities = self._get_major_cities_nearby(current_city)
            for city in major_cities:
                if city not in checked_cities:
                    result = await self._check_city_availability(city, filters)
                    if result:
                        nearby.append(result)
                    checked_cities.add(city)
                    if len(nearby) >= 5:
                        break

        # Sort by driver count and distance
        nearby.sort(key=lambda x: (-x["driver_count"], x.get("distance_score", 999)))

        return nearby[:10]  # Return top 10 options

    def _get_cluster_cities(self, city: str) -> List[str]:
        """
        Get cities in the same cluster.

        Args:
            city: City name

        Returns:
            List of cluster cities
        """
        city_lower = city.lower()
        for cluster in self.CITY_CLUSTERS.values():
            if city_lower in cluster["cities"]:
                return cluster["cities"]
        return []

    async def _get_nearby_cities_from_llm(self, city: str) -> List[str]:
        """
        Get nearby cities using LLM knowledge.

        Args:
            city: City name

        Returns:
            List of nearby cities
        """
        prompt = f"""List 5-7 cities near {city} in India.

            Requirements:
                1. Include both major cities and smaller towns
                2. Within reasonable travel distance (up to 200km)
                3. Mix of different directions from {city}
                4. Return as comma-separated list
                5. Use common spellings

                Return only city names, nothing else.
        """

        try:
            response = await self.llm.ainvoke(prompt)
            cities = [c.strip().lower() for c in response.content.split(",")]
            return cities[:7]
        except Exception as e:
            logger.error(f"LLM city suggestion failed: {e}")
            return []

    def _get_major_cities_nearby(self, city: str) -> List[str]:
        """
        Get major cities as fallback options.

        Args:
            city: Current city

        Returns:
            List of major cities
        """
        # Major cities by region
        major_cities = {
            "north": ["delhi", "gurgaon", "noida", "chandigarh", "jaipur"],
            "west": ["mumbai", "pune", "ahmedabad", "surat", "nashik"],
            "south": ["bangalore", "chennai", "hyderabad", "kochi", "coimbatore"],
            "east": ["kolkata", "bhubaneswar", "guwahati", "patna", "ranchi"],
        }

        # Try to determine region and return relevant cities
        # This is simplified - in production, use proper geographic data
        all_major = []
        for cities in major_cities.values():
            all_major.extend(cities)

        return [c for c in all_major if c.lower() != city.lower()][:10]

    async def _check_city_availability(
        self, city: str, filters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if city has drivers matching filters.

        Args:
            city: City to check
            filters: Active filters

        Returns:
            City info with driver count or None
        """
        try:
            # First check without filters to get total
            result = await self.driver_tools.search_drivers_tool(
                city=city, page=1, limit=1, use_cache=True
            )

            if not result["success"] or result["total"] == 0:
                return None

            total_drivers = result["total"]

            # If filters active, check with filters
            filtered_drivers = total_drivers
            if filters:
                # Apply filters through search
                filter_params = {"city": city, "page": 1, "limit": 1}
                for k, v in filters.items():
                    filter_params[k] = v

                filter_result = await self.driver_tools.search_drivers_tool(
                    **filter_params
                )
                if filter_result["success"]:
                    filtered_drivers = filter_result["total"]

            if filtered_drivers > 0:
                return {
                    "city": city,
                    "driver_count": filtered_drivers,
                    "total_drivers": total_drivers,
                    "matches_filters": filtered_drivers > 0,
                }

            return None

        except Exception as e:
            logger.warning(f"Error checking {city}: {e}")
            return None

    def _group_cities_by_proximity(
        self, current_city: str, cities: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group cities by proximity/region.

        Args:
            current_city: Current city
            cities: List of city data

        Returns:
            Grouped cities
        """
        grouped = {"same_region": [], "nearby": [], "other_options": []}

        # Check if current city is in a cluster
        current_cluster = None
        for cluster_name, cluster_data in self.CITY_CLUSTERS.items():
            if current_city.lower() in cluster_data["cities"]:
                current_cluster = cluster_data["cities"]
                break

        for city_data in cities:
            city_name = city_data["city"].lower()

            if current_cluster and city_name in current_cluster:
                grouped["same_region"].append(city_data)
            elif len(grouped["nearby"]) < 3:
                grouped["nearby"].append(city_data)
            else:
                grouped["other_options"].append(city_data)

        # Remove empty groups
        return {k: v for k, v in grouped.items() if v}

    async def _generate_city_suggestions_message(
        self,
        current_city: str,
        grouped_cities: Dict[str, List[Dict[str, Any]]],
        filters: Dict[str, Any],
        language: str,
    ) -> str:
        """
        Generate message with city suggestions.

        Args:
            current_city: Current city with no results
            grouped_cities: Cities grouped by proximity
            filters: Active filters
            language: Response language

        Returns:
            Suggestion message
        """
        # Flatten for easier access
        all_cities = []
        for group, cities in grouped_cities.items():
            for city in cities:
                city["group"] = group
                all_cities.append(city)

        filter_summary = ""
        if filters:
            filter_parts = [f"{k}: {v}" for k, v in filters.items()]
            filter_summary = f"Filters: {', '.join(filter_parts)}"

        prompt = f"""Generate a helpful message suggesting nearby cities with drivers.

                Context:
                    - No drivers found in: {current_city}
                    - {filter_summary}
                    - Language: {language}

                Nearby cities with drivers:
                    {all_cities[:5]}  # Show top 5

                Requirements:
                    1. Present cities clearly with driver counts
                    2. Group by proximity if applicable (same region, nearby, others)
                    3. If filters active, mention which cities have filtered matches
                    4. Make it easy to choose (use numbers if ≤5 cities)
                    5. Suggest user can search in any of these
                    6. Keep positive and helpful tone
                    7. Use {language} language

                    Generate only the message.
                """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate message: {e}")
            # Fallback
            if language == "hindi":
                msg = f"{current_city} में ड्राइवर नहीं मिले। यहाँ आस-पास के विकल्प हैं:\n"
                for i, city in enumerate(all_cities[:3]):
                    msg += f"{i + 1}. {city['city']} - {city['driver_count']} ड्राइवर\n"
                return msg
            else:
                msg = f"No drivers in {current_city}. Here are nearby options:\n"
                for i, city in enumerate(all_cities[:3]):
                    msg += f"{i + 1}. {city['city'].title()} - {
                        city['driver_count']
                    } drivers\n"
                msg += "\nWhich city would you like to search in?"
                return msg

    async def _generate_no_alternatives_message(self, city: str, language: str) -> str:
        """
        Generate message when no nearby alternatives found.

        Args:
            city: Current city
            language: Response language

        Returns:
            No alternatives message
        """
        prompt = f"""Generate a message when no nearby cities have drivers.

                Context:
                    - Searched city: {city}
                    - No nearby alternatives found
                    - Language: {language}

                Requirements:
                    1. Acknowledge the situation
                    2. Suggest trying major cities
                    3. Mention service may expand
                    4. Stay helpful and positive
                    5. Use {language} language

                Generate only the message.
            """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception:
            if language == "hindi":
                return (
                    f"{city} या आस-पास के शहरों में ड्राइवर उपलब्ध नहीं हैं। कृपया बड़े शहरों में खोजें।"
                )
            else:
                return f"No drivers available in {city} or nearby areas. Try searching in major cities like Delhi, Mumbai, or Bangalore."
