from langchain.tools import Tool, StructuredTool
import time
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, TypedDict
from src.models.drivers_model import Driver_Model, Premium_driver_model
from src.services.api_service import APIService
import asyncio


class Drive_search_inputs(BaseModel):
    """Input schema for driver search"""

    city: str = Field(description="City name user want to travel from")
    page: int = Field(default=0, description="Page number for pagination")
    limit: int = Field(
        default=10,
        description="Limit for fetching number of premium\
                drivers at once",
    )
    timestamp: int = Field(
        description="timesatmp for verifying that request is not too old"
    )
    preference: Union[str, Dict] = Field(
        default="",
        description="Find premeium drivers based on user's preference, we'll\
                empty it for now",
    )


class Driver_filter_inputs(BaseModel):
    """Input schema for driver filtering"""

    drivers: List[Dict[str, Any]] = Field(
        description="List of drivers to\
            filter from"
    )
    filters: Dict[str, Union[str, bool]] = Field(
        description="Filter criteria set by user"
    )


class DriverCacheEntry(TypedDict):
    data: List[Driver_Model]
    expires: datetime


class PremiumDriversCacheEntry(TypedDict):
    data: List[Premium_driver_model]
    expires: datetime


class DriverTools:
    """It is a collection of all the tools related to drivers"""

    def __init__(self, user_session_id: str):
        self.api_service = APIService()
        self.user_session_id = user_session_id
        self._premium_drivers_cache: Dict[str, PremiumDriversCacheEntry] = {}
        self._drivers_cache: Dict[str, DriverCacheEntry] = {}

    async def fetch_drivers(
        self,
        city: str,
        page: int,
        limit: int = 10,
        timestamp: int = int(time.time()),
        preference: str = "",
    ) -> List[Premium_driver_model]:
        """Fetch premium drivers and parse it according\
                to our Premium_driver_model"""

        cache_key = f"Premium_driver_{self.user_session_id}_{city}_{page}"
        if cache_key in self._premium_drivers_cache:
            cached_data = self._premium_drivers_cache[cache_key]
            if cached_data.get("expires") > datetime.now():
                return cached_data.get("data")

        raw_response = await self.api_service.get_premium_driver_by_location(
            city, page, limit, timestamp, preference
        )

        if not raw_response or not raw_response.get("success"):
            print("API call failed or returned unsuccessful.")
            return []

        raw_driver_list = raw_response.get("data", [])

        try:
            parsed_drivers = [
                Premium_driver_model.model_validate(driver_data)
                for driver_data in raw_driver_list
            ]

            # Store parsed drivers in cache
            self._premium_drivers_cache[cache_key] = {
                "data": parsed_drivers,
                "expires": datetime.now() + timedelta(minutes=5),
            }

            # Create async task to fetch detailed driver data
            asyncio.create_task(
                self._fetch_and_cache_driver_detail(parsed_drivers, city)
            )

            return parsed_drivers

        except Exception as e:
            print(f"Bazinga! A validation error occurred: {e}")
            return []

    async def _fetch_and_cache_driver_detail(
        self, premium_drivers: List[Premium_driver_model], city: str
    ):
        tasks = []
        # Create a mapping of driver_id to premium driver for easy lookup
        driver_map = {driver.id: driver for driver in premium_drivers}

        for driver in premium_drivers:
            tasks.append(self.api_service.get_partner_data(driver.id, int(time.time())))

        # Wait for all API responses
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_drivers: List[Driver_Model] = []

        for res in results:
            if isinstance(res, Exception):
                print("Error in fetching driver:", res)
                continue

            try:
                if not res or not res.get("success"):
                    print("Invalid response:", res)
                    continue

                driver_data = res.get("data", {})
                driver_uid = driver_data.get("uid")

                # Find the corresponding premium driver info
                existing_info = driver_map.get(driver_uid)

                if not existing_info:
                    print(f"No existing info found for driver {driver_uid}")
                    continue

                print("existing_info: \n", existing_info)

                # Add existing_info to driver_data
                driver_data["existing_info"] = existing_info

                validated = Driver_Model.model_validate(driver_data)
                valid_drivers.append(validated)
            except Exception as e:
                print("Validation error:", e)
                print("Driver data:", driver_data)

        self._drivers_cache[f"{self.user_session_id}_{city}"] = {
            "data": valid_drivers,
            "expires": datetime.now() + timedelta(minutes=5),
        }

    async def get_all_drivers_from_cache(
        self, city: str
    ) -> Optional[List[Driver_Model]]:
        cache_key = f"{self.user_session_id}_{city}"

        cache_entry = self._drivers_cache.get(cache_key)

        if not cache_entry:
            print("🚫 No cache found for city:", city)
            return None

        expires_at = cache_entry["expires"]
        if expires_at < datetime.now():
            print("⚠️ Cache expired for city:", city)
            del self._drivers_cache[cache_key]  # Fixed typo
            return None

        print("✅ Cache hit for city:", city)
        return cache_entry["data"]


__all__ = ["DriverTools"]
