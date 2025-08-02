import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from src.models.drivers_model import DriverModel, APIResponse

logger = logging.getLogger(__name__)


class DriversAPIClient:
    """Client for interacting with the Premium Drivers API"""

    def __init__(self, session_id: str, cache_duration_minutes: int = 5):
        self.base_url = "https://us-central1-cabswale-ai.cloudfunctions.net"
        self.endpoint = "/cabbot-botApiGetPremiumDrivers"
        self.client = httpx.AsyncClient(timeout=30.0)
        self._cache = {}  # Simple in-memory cache for now
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.session_id = session_id

    def _generate_cache_key(self, city: str, page: int) -> str:
        """Generate a cache key from parameters"""
        return f"{self.session_id}_{city}_{page}"

    async def _get_from_cache(self, cache_key: str) -> APIResponse:
        """Get data from cache if not expired"""
        # print("\ngetting from cache.....\n")

        if cache_key in self._cache:
            cached_data = self._cache.get(cache_key)
            if cached_data.get("expires") > datetime.now():
                logger.info(f"Cache hit for key: {cache_key}")
                return cached_data.get("data")
            else:
                del self._cache[cache_key]
        return None

    def _save_to_cache(self, cache_key: str, data: APIResponse):
        # print("\nSaving to cache.....\n")
        """Save data to cache with expiration"""
        self._cache[cache_key] = {
            "data": data,
            "expires": datetime.now() + self.cache_duration,
        }
        logger.info(f"Cached data for key: {cache_key}")

    async def _get_driver_detail(self, cache_key: str, driverId: str) -> DriverModel:
        drivers_from_cache = await self._get_from_cache(cache_key)
        driver_data: DriverModel = {}
        for driver in drivers_from_cache.data:
            if driverId == driver.id:
                driver_data: DriverModel = DriverModel.model_validate(driver)
                break
        return driver_data

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
    ) -> APIResponse:
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
        if vehicle_types:
            params["vehicleTypes"] = ",".join(vehicle_types)
        if gender:
            params["gender"] = gender
        if min_age is not None:
            params["minAge"] = min_age
        if max_age is not None:
            params["maxAge"] = max_age
        if is_pet_allowed is not None:
            params["isPetAllowed"] = str(is_pet_allowed).lower()
        if min_connections is not None:
            params["minConnections"] = min_connections
        if min_experience is not None:
            params["minExperience"] = min_experience
        if languages:
            params["verifiedLanguages"] = ",".join(languages)
        if profile_verified is not None:
            params["profileVerified"] = profile_verified
        if married:
            params["married"] = married

        # Add any custom filters
        if custom_filters:
            params.update(custom_filters)

        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(city, page)
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                return cached_data

        try:
            # Make API request
            url = f"{self.base_url}{self.endpoint}"
            logger.info(
                f"Fetching premium drivers from: {url}\
                    with params: {params}"
            )
            print("Calling api with params -> \n", params)

            response = await self.client.get(url, params=params)
            response.raise_for_status()

            data: APIResponse = APIResponse.model_validate(response.json())
            # print("\nGOT DATA\n", data)

            # Cache successful response
            if use_cache and data.success:
                cache_key = self._generate_cache_key(city, page)
                self._save_to_cache(cache_key, data)

            return data

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error {e.response.status_code}:\
                    {e.response.text}"
            )
            return {
                "success": False,
                "message": f"HTTP error {e.response.status_code}:\
                        {e.response.text}",
            }
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            return {"success": False, "message": f"Request failed: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"success": False, "message": f"Unexpected error: {str(e)}"}

    def clear_cache(self, city: Optional[str] = None):
        """Clear cache for specific city or all cache"""
        if city:
            # Clear cache for specific city
            keys_to_remove = [k for k in self._cache.keys() if f"city={city}" in k]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Cleared cache for city: {city}")
        else:
            # Clear all cache
            self._cache.clear()
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
