import os
import asyncio
import time
from dotenv import load_dotenv
from src.lngraph.tools.driver_tools import DriverTools

load_dotenv()


async def main():
    tools = DriverTools("xnxx69")
    await tools.fetch_drivers("Delhi", 1, 1, int(time.time()), "")
    await asyncio.sleep(2)
    data = await tools.get_all_drivers_from_cache("Delhi")
    print(data)
    # print(os.environ["GET_PREMIUM_DRIVERS_URL"])


if __name__ == "__main__":
    asyncio.run(main())
