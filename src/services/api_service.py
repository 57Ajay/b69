import httpx
from typing import Dict, Any, List, Optional, Union
from datetime import timedelta
import logging
from src.models.drivers_model import DriverModel, APIResponse
from src.services.cache_service import RedisService

logger = logging.getLogger(__name__)


class DriversAPIClient:
    """Client for interacting with the Premium Drivers API"""

    def __init__(self, session_id: str, redis_service: RedisService, cache_duration_minutes: int = 10):
        self.base_url = "https://us-central1-cabswale-ai.cloudfunctions.net"
        self.endpoint = "/cabbot-botApiGetPremiumDrivers"
        self.client = httpx.AsyncClient(timeout=30.0)
        self.redis_service = redis_service
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.session_id = session_id

    def _generate_cache_key(self, city: str, page: int) -> str:
        """Generate a cache key from parameters"""
        return f"{self.session_id}_{city}_{page}"

    async def _get_from_cache(self, cache_key: str) -> Union[Dict[str, Any], None]:
        """Get data from cache if not expired"""
        cached_data = await self.redis_service.get(cache_key)
        if cached_data:
            logger.info(f"Cache hit for key: {cache_key}")
            return cached_data
        return None

    async def _save_to_cache(self, cache_key: str, data: APIResponse):
        """Save data to cache with expiration"""
        await self.redis_service.set(
            cache_key, data.model_dump(by_alias=True), expiration_seconds=int(self.cache_duration.total_seconds())
        )
        logger.info(f"Cached data for key: {cache_key}")

    async def _get_driver_detail(self, cache_key: str, driverId: str) -> DriverModel:
        """Get specific driver details from cache"""
        drivers_from_cache = await self._get_from_cache(cache_key)
        if not drivers_from_cache:
            raise ValueError(f"No cached data found for cache key: {cache_key}")

        # Parse the cached response
        api_response = APIResponse.model_validate(drivers_from_cache)

        # Find the specific driver
        for driver in api_response.data:
            if driverId == driver.id:
                return driver

        raise ValueError(f"Driver with ID {driverId} not found in cached data")

    def _build_driver_filters(
        self,
        vehicle_types: Optional[List[str]],
        gender: Optional[str],
        min_age: Optional[int],
        max_age: Optional[int],
        is_pet_allowed: Optional[bool],
        min_connections: Optional[int],
        min_experience: Optional[int],
        languages: Optional[List[str]],
        profile_verified: Optional[bool],
        married: Optional[bool],
        custom_filters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build filter dictionary for API request"""
        filters: Dict[str, Any] = {}
        logger.debug("Building driver filters")

        optional_list_filters = {
            "vehicleTypes": vehicle_types,
            "verifiedLanguages": languages,
        }

        for key, val in optional_list_filters.items():
            if val:
                filters[key] = ",".join(val)

        optional_simple_filters = {
            "gender": gender,
            "minAge": min_age,
            "maxAge": max_age,
            "minConnections": min_connections,
            "minExperience": min_experience,
            "profileVerified": profile_verified,
            "married": married,
        }

        for key, val in optional_simple_filters.items():
            if val is not None:
                filters[key] = val

        if is_pet_allowed is not None:
            filters["isPetAllowed"] = str(is_pet_allowed).lower()

        if custom_filters:
            filters.update(custom_filters)

        return filters

    async def get_drivers(
        self,
        city: str,
        page: int = 1,
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
        custom_filters: Optional[Dict[str, Any]] = None,
        married: Optional[bool] = None,
        use_cache: bool = True,
    ) -> Dict[str, Union[str, int, bool, APIResponse]]:
        """
        Fetch premium drivers with various filters

        Args:
            city: City name to search in
            page: Page number (default 1)
            limit: Results per page (default 10, max 100)
            radius: Search radius in km (default 100)
            search_strategy: "city", "geo", or "hybrid" (default)
            sort_by: Sort field and order (default "lastAccess:desc")
            vehicle_types: List of vehicle types to filter
            gender: Filter by gender ("male" or "female")
            min_age: Minimum age filter
            max_age: Maximum age filter
            is_pet_allowed: Filter drivers who allow pets
            min_connections: Minimum connections required
            min_experience: Minimum years of experience
            languages: List of languages to filter by
            profile_verified: Filter by profile verification status
            custom_filters: Additional custom filters as dict
            use_cache: Whether to use cache (default True)

        Returns:
            API response as dictionary or error response
        """
        # Build query parameters
        params = {
            "city": city,
            "page": page,
            "limit": limit,
            "radius": radius,
            "searchStrategy": search_strategy,
            "sortBy": sort_by,
        }

        # Add optional filters
        filters = self._build_driver_filters(
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
        )
        params.update(filters)
        # print("\n\nPARAMS:\n\n", params)
        logger.debug(f"Filters applied: {filters}")

        # Check cache first if enabled
        if use_cache:
            cache_key = self._generate_cache_key(city, page)
            cached_data = await self._get_from_cache(cache_key)

            if cached_data:
                # Convert cached data to APIResponse
                api_response = APIResponse.model_validate(cached_data)
                return {"success": True, "data": api_response}

        try:
            url = f"{self.base_url}{self.endpoint}"
            logger.info(f"Fetching premium drivers from: {url} with params: {params}")

            response = await self.client.get(url, params=params)
            response.raise_for_status()

            data: APIResponse = APIResponse.model_validate(response.json())

            # Cache successful response
            if use_cache and data.success:
                cache_key = self._generate_cache_key(city, page)
                await self._save_to_cache(cache_key, data)

            return {"success": True, "data": data}

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
            }
        except httpx.RequestError as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg}

    async def clear_cache(self, city: Optional[str] = None):
        """Clear cache for specific city or all cache"""
        if city:
            # Clear cache for specific city
            keys_to_remove = [k async for k in self.redis_service.redis_client.scan_iter(f"*_{city}_*")]
            if keys_to_remove:
                await self.redis_service.redis_client.delete(*keys_to_remove)
            logger.info(f"Cleared cache for city: {city}")
        else:
            # Clear all cache
            await self.redis_service.clear_all()
            logger.info("Cleared all cache")

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


__all__ = ["DriversAPIClient"]
