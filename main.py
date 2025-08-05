from src.lngraph.tools.driver_tools import DriverTools
from src.services.api_service import DriversAPIClient
import asyncio


async def main():
    api_client = DriversAPIClient("1", 2)
    driver_tools = DriverTools(api_client=api_client)
    print("drivers: ", await driver_tools.search_drivers_tool("delhi", 1, 1))


if __name__ == "__main__":
    asyncio.run(main())
