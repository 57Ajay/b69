import httpx
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta


class APIService:
    def __init__(self):
        self.base_url = "https://us-central1-cabswale-ai.cloudfunctions.net"
        self.client = httpx.AsyncClient(timeout=30.0)
        self._cache = {}  # Simple in-memory cache for now, i plan to use redis with it, but for now it is ok

    async def get_premium_driver_by_location(
        self,
        city: str,
        page: int = 0,
        limit: int = 10,
        timestamp: int = int(time.time()),
        preference: Union[str, Dict] = "",
    ) -> List[Dict[str, Any]]:
        """Fetches drivers by location based on pagination"""
        cache_key = f"drivers_{city}_{page}_{limit}"

        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            if cached_data["expires"] > datetime.now():
                return cached_data
        try:
            response = await self.client.post(
                f"{self.base_url}/typesense-getPartnersByLocation",
                json={
                    "city": city,
                    "page": page,
                    "limit": limit,
                    "timestamp": timestamp,
                    "preference": preference,
                },
            )
            response.raise_for_status()
            data = response.json()
            self._cache[cache_key] = {
                "data": data,
                "expires": datetime.now() + timedelta(minutes=5),
            }
            return data
        except httpx.HTTPError as e:
            print(f"Error fetching drivers: {e}")
            return []

    async def get_partner_data(
        self, partner_id: str, timestamp: int
    ) -> Optional[Dict[str, Any]]:
        """Fetch detailed partner/driver data"""

        try:
            response = await self.client.post(
                f"{self.base_url}/partners-getPartnerData",
                json={"partnerId": partner_id, "timestamp": timestamp},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"Error fetching partner data: {e}")
            return None

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


__all__ = ["APIService"]
