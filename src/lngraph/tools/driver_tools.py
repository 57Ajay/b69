"""
Driver tools for the cab booking bot.
Simple tools that the LLM can use to interact with the API.
"""

from typing import Dict, List, Optional, Any, Union
from langchain_core.tools import tool
import logging
from src.services.api_service import DriversAPIClient
from src.models.tool_model import (
    SearchDriversInput,
    DriverInfoInput,
    FilterDriverInput,
    BookDriverInput,
)
from src.models.drivers_model import DriverModel, APIResponse

logger = logging.getLogger(__name__)


class DriverTools:
    """Tools for driver operations"""

    def __init__(self, api_client: DriversAPIClient):
        """
        Initialize driver tools.

        Args:
            api_client: Instance of DriversAPIClient
        """
        self.api_client = api_client

        # Create bound tools with the API client
        self.search_drivers_tool = self._create_search_drivers_tool()
        self.get_driver_info_tool = self._create_get_driver_info_tool()
        self.get_drivers_with_user_filter_via_cache_tool = self._create_filter_drivers_tool()
        self.book_or_confirm_ride_with_driver = self._create_book_driver_tool()

    def _create_search_drivers_tool(self):
        """Create the search drivers tool with bound API client"""
        api_client = self.api_client

        @tool(
            description="""
        Comprehensive driver search with advanced filtering and pagination support.
        This is the primary tool for finding drivers based on various criteria:

        **Core Functionality:**
        - Full-featured driver search with 15+ filter options
        - Intelligent caching for improved performance
        - Pagination support for large result sets
        - Multiple search strategies (city, geo-location, hybrid)

        **Filter Categories:**
        1. **Vehicle Preferences**: Types (sedan, SUV, etc.), specific models
        2. **Demographics**: Age range, gender, marital status
        3. **Service Options**: Pet-friendly, language preferences
        4. **Professional Criteria**: Experience, completed rides, verification
        5. **Location**: City-based or radius-based search

        **Smart Features:**
        - Hybrid search combines city and geo-location for best results
        - Customizable sorting (by last active, rating, experience)
        - Supports complex queries with multiple filters
        - Returns metadata including total count and pagination info

        **Example Queries:**
        - "Find top-rated SUV drivers in Mumbai"
        - "Show experienced female drivers who speak English in Delhi"
        - "Get pet-friendly drivers with 5+ years experience"

        **Response includes:**
        - Matching drivers with complete profiles
        - Total count and pagination details
        - Applied filters for transparency
        - Success/error status
        """,
            args_schema=SearchDriversInput,
        )
        async def search_drivers_tool(
            city: str,
            page: int,
            limit: int = 10,
            radius: int = 100,
            search_strategy: str = "hybrid",
            sort_by: str = "lastAccess:desc",
            vehicle_types: Optional[List[str]] = None,
            gender: Optional[str] = None,
            min_age: Optional[int] = None,
            max_age: Optional[int] = None,
            is_pet_allowed: Optional[bool] = None,
            min_connections: Optional[int] = None,
            min_experience: Optional[int] = None,
            languages: Optional[List[str]] = None,
            profile_verified: Optional[bool] = None,
            married: Optional[bool] = None,
            custom_filters: Optional[Dict[str, Any]] = None,
            use_cache: bool = True,
        ) -> Dict[str, Union[str, bool, List[DriverModel], int, Dict[str, Any]]]:
            try:
                # Call API with parameters
                result = await api_client.get_drivers(
                    city=city,
                    page=page,
                    limit=limit,
                    radius=radius,
                    search_strategy=search_strategy,
                    sort_by=sort_by,
                    vehicle_types=vehicle_types,
                    gender=gender,
                    min_age=min_age,
                    max_age=max_age,
                    is_pet_allowed=is_pet_allowed,
                    min_connections=min_connections,
                    min_experience=min_experience,
                    languages=languages,
                    profile_verified=profile_verified,
                    married=married,
                    custom_filters=custom_filters,
                    use_cache=use_cache,
                )
                if not result.get("success", False):
                    return {
                        "success": False,
                        "error": str(result.get("message", "Failed to get drivers")),
                    }

                drivers = APIResponse.model_validate(result.get("data")).data

                return {
                    "success": True,
                    "drivers": drivers,
                    "count": len(drivers),
                    "total": APIResponse.model_validate(result.get("data")).pagination.total,
                    "filters": APIResponse.model_validate(result.get("data")).search.filters,
                    "has_more": APIResponse.model_validate(result.get("data")).pagination.has_more,
                    "page": page,
                }

            except Exception as e:
                logger.error(f"Error searching drivers: {str(e)}")
                return {
                    "success": False,
                    "msg": "Failed to search drivers for your city, try again later",
                    "error": str(e),
                }

        return search_drivers_tool

    def _create_get_driver_info_tool(self):
        """Create the get driver info tool with bound API client"""
        api_client = self.api_client

        @tool(
            description="""
        Retrieves detailed information for a specific driver from cache. Optimized for quick
        lookups when users want to know more about a particular driver:

        **Primary Use Cases:**
        - User clicks on a driver from search results
        - Following up on a previously viewed driver
        - Getting updated details for a known driver ID

        **Required Information:**
        - Driver ID (unique identifier)
        - City and page (for efficient cache retrieval)

        **Returns:**
        - Complete driver profile including:
          - Personal details (name, age, experience)
          - Vehicle information and images
          - Ratings and reviews
          - Availability status
          - Contact preferences
          - Service options (pet-friendly, languages, etc.)

        **Performance Notes:**
        - Cache-based retrieval ensures instant response
        - No API call needed if driver exists in cache
        - Ideal for drill-down scenarios after initial search

        **Example Usage:**
        - "Show me more about driver DRV123456"
        - "Get details for the third driver from the previous search"
        - "What vehicles does driver DRV789012 have?"
        """,
            args_schema=DriverInfoInput,
        )
        async def get_driver_info_tool(
            city: str, page: int, driverId: str
        ) -> Dict[str, Union[DriverModel, bool, str, Exception]]:
            try:
                driver: DriverModel = await api_client._get_driver_detail(
                    api_client._generate_cache_key(city=city, page=page),
                    driverId=driverId,
                )
                return {"success": True, "driver": driver}
            except Exception as e:
                return {
                    "success": False,
                    "msg": "Failed to get Driver Information",
                    "error": e,
                }

        return get_driver_info_tool

    def _create_filter_drivers_tool(self):
        """Create the filter drivers tool with bound API client"""
        api_client = self.api_client

        @tool(
            description="""
        Retrieves drivers from cache based on user-specified filter criteria. This tool provides
        intelligent driver matching with fallback mechanisms:

        **Primary Function:**
        - Searches cached driver data for matches based on multiple filter criteria
        - Supports complex filtering including vehicle types, demographics, experience, and availability

        **Smart Fallback Logic:**
        - If fewer than 5 drivers match the criteria, automatically triggers a fresh API search
        - api search will be done again by `search_drivers` tool with next `page` value
        - Ensures users always get sufficient options while leveraging cache for performance

        **Filter Capabilities:**
        - Vehicle type matching (supports multiple types with OR logic)
        - Demographic filters (age range, gender, marital status)
        - Service preferences (pet-friendly, handicapped assistance, event availability)
        - Professional criteria (experience, connections, verification status)

        **Use Cases:**
        - "Find female drivers in Delhi who allow pets"
        - "Get experienced drivers with sedan or SUV in Mumbai"
        - "Show verified drivers available for wedding events"

        **Performance Note:**
        Cache-first approach ensures fast response times for common queries while
        maintaining result quality through automatic API fallback when needed.
        """,
            args_schema=FilterDriverInput,
        )
        async def get_drivers_with_user_filter_via_cache_tool(
            city: str, page: int, filter_obj: Dict[str, Any]
        ) -> Dict[str, Union[List[DriverModel], int, str, Exception]]:
            try:
                ALLOWED_FILTER_KEYS = {
                    "vehicle_types",
                    "gender",
                    "min_age",
                    "max_age",
                    "is_pet_allowed",
                    "min_connections",
                    "min_experience",
                    "languages",
                    "profile_verified",
                    "married",
                    "allow_handicapped_persons",
                    "available_for_customers_personal_car",
                    "available_for_driving_in_event_wedding",
                    "available_for_part_time_full_time",
                }

                raw_drivers_response = await api_client._get_from_cache(
                    api_client._generate_cache_key(city=city, page=page)
                )
                if raw_drivers_response is None:
                    return {"success": False, "msg": "No drivers found"}

                raw_drivers: List[DriverModel] = APIResponse.model_validate(raw_drivers_response).data

                valid_filter_obj = {
                    k: v for k, v in filter_obj.items() if k in ALLOWED_FILTER_KEYS
                }

                def matches_filter(driver: DriverModel, key: str, value: Any) -> bool:
                    if key == "vehicle_types":
                        driver_vehicle_types = [
                            vehicle.vehicle_type for vehicle in driver.verified_vehicles
                        ]
                        if isinstance(value, list):
                            return any(vtype in driver_vehicle_types for vtype in value)
                        else:
                            return value in driver_vehicle_types

                    # Handler for min/max filters
                    if key.startswith("min_"):
                        attr_name = key[4:]
                        driver_value = getattr(driver, attr_name, None)
                        return driver_value is not None and driver_value >= value

                    elif key.startswith("max_"):
                        attr_name = key[4:]
                        driver_value = getattr(driver, attr_name, None)
                        return driver_value is not None and driver_value <= value

                    else:
                        driver_value = getattr(driver, key, None)
                        return driver_value == value

                filtered_drivers = [
                    driver
                    for driver in raw_drivers
                    if all(
                        matches_filter(driver, k, v) for k, v in valid_filter_obj.items()
                    )
                ]

                validated_drivers: List[DriverModel] = [
                    DriverModel.model_validate(driver) for driver in filtered_drivers
                ]
                return {
                    "success": True,
                    "filtered_drivers": validated_drivers,
                    "total": len(validated_drivers),
                }
            except Exception as e:
                return {
                    "success": False,
                    "msg": "Failed to apply the filter, Please choose appropriate filters",
                    "error": e,
                }

        return get_drivers_with_user_filter_via_cache_tool

    def _create_book_driver_tool(self):
        """Create the book driver tool with bound API client"""
        api_client = self.api_client

        @tool(
            description="""
        Retrieves drivers from cache when User asks to book ride with
        the driver. This tool provides is used when user asks for ride
        with a specific driver

        **Primary Function:**
        - Get's the driver user asked to book ride with.
        - Give user Driver's Name, Profile url and contact number

        **Smart Fallback Logic:**
        - If user asks for Driver who is not in the List or some issue occures,
            Simply return user that there is some error finding that driver,
            and tell user he can choose another driver

        **Use Cases:**
        - "Book my ride with Ramesh"
        - "Confirm booking with Suresh"
        - "Book him for me ( here model will get the driver in context\
                and choose that driver and show his Info) "

        **Performance Note:**
        Cache-first approach ensures fast response times for common queries while
        """,
            args_schema=BookDriverInput,
        )
        async def book_or_confirm_ride_with_driver(
            city: str, page: int, driverId: str
        ) -> Dict[str, Union[bool, str, Exception]]:
            try:
                driver: DriverModel = await api_client._get_driver_detail(
                    api_client._generate_cache_key(city=city, page=page),
                    driverId=driverId,
                )

                return {
                    "success": True,
                    "Driver Name": driver.name,
                    "Profile": driver.profile_url,
                    "PhoneNo.": driver.phone_no,
                }
            except Exception as e:
                return {"success": False, "msg": "Failed to get the driver", "error": e}

        return book_or_confirm_ride_with_driver
