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
        self.search_drivers_tool = self._create_search_drivers_tool()
        self.get_driver_info_tool = self._create_get_driver_info_tool()
        self.book_or_confirm_ride_with_driver = self._create_book_driver_tool()

    def _create_search_drivers_tool(self):
        """Create the search drivers tool with bound API client"""
        api_client = self.api_client

        @tool(
            description="""
        Comprehensive driver search with advanced filtering and pagination support.
        This is the primary tool for finding drivers based on various criteria.
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
        lookups when users want to know more about a particular driver.
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

    def _create_book_driver_tool(self):
        """Create the book driver tool with bound API client"""
        api_client = self.api_client

        @tool(
            description="""
        Retrieves drivers from cache when User asks to book ride with
        the driver. This tool provides is used when user asks for ride
        with a specific driver
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
