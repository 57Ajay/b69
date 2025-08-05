from src.services.cache_service import RedisService
from src.lngraph.tools.driver_tools import DriverTools
from src.services.api_service import DriversAPIClient
import asyncio


async def main():
    redis_service = RedisService(host="localhost", port=6379)
    api_client = DriversAPIClient("1", redis_service, 2)
    driver_tools = DriverTools(api_client=api_client)
    print("drivers: ", await driver_tools.search_drivers_tool("delhi", 2, 100))


if __name__ == "__main__":
    asyncio.run(main())
