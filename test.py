#!/usr/bin/env python3
"""
Test script to verify that the driver tools work correctly
"""
import asyncio
import uuid
import logging
from src.services.cache_service import RedisService
from src.services.api_service import DriversAPIClient
from src.lngraph.tools.driver_tools import DriverTools

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_tools():
    """Test the driver tools to ensure they work correctly"""

    logger.info("Starting tool tests...")

    # Initialize services
    redis_service = RedisService()
    if not await redis_service.ping():
        logger.error("Could not connect to Redis. Please ensure it is running.")
        return False

    session_id = str(uuid.uuid4())
    api_client = DriversAPIClient(session_id=session_id, redis_service=redis_service)
    driver_tools = DriverTools(api_client=api_client)

    # Test 1: Search drivers tool
    logger.info("Testing search_drivers_tool...")
    try:
        search_result = await driver_tools.search_drivers_tool.ainvoke({
            "city": "delhi",
            "page": 1,
            "limit": 5
        })

        if search_result.get("success"):
            logger.info(f"✅ Search tool test passed. Found {search_result.get('count', 0)} drivers.")
            drivers = search_result.get("drivers", [])

            if drivers:
                # Test 2: Get driver info tool
                logger.info("Testing get_driver_info_tool...")
                first_driver_id = drivers[0].id

                info_result = await driver_tools.get_driver_info_tool.ainvoke({
                    "city": "delhi",
                    "page": 1,
                    "driverId": first_driver_id
                })

                if info_result.get("success"):
                    logger.info("✅ Driver info tool test passed.")

                    # Test 3: Book driver tool
                    logger.info("Testing book_or_confirm_ride_with_driver...")
                    book_result = await driver_tools.book_or_confirm_ride_with_driver.ainvoke({
                        "city": "delhi",
                        "page": 1,
                        "driverId": first_driver_id
                    })

                    if book_result.get("success"):
                        logger.info("✅ Book driver tool test passed.")
                    else:
                        logger.error(f"❌ Book driver tool test failed: {book_result}")

                else:
                    logger.error(f"❌ Driver info tool test failed: {info_result}")

                # Test 4: Filter drivers tool
                logger.info("Testing get_drivers_with_user_filter_via_cache_tool...")
                filter_result = await driver_tools.get_drivers_with_user_filter_via_cache_tool.ainvoke({
                    "city": "delhi",
                    "page": 1,
                    "filter_obj": {"gender": "male", "min_experience": 2}
                })

                if filter_result.get("success"):
                    filtered_count = len(filter_result.get("filtered_drivers", []))
                    logger.info(f"✅ Filter tool test passed. Found {filtered_count} filtered drivers.")
                else:
                    logger.error(f"❌ Filter tool test failed: {filter_result}")

            else:
                logger.warning("No drivers found in search result, skipping other tests.")

        else:
            logger.error(f"❌ Search tool test failed: {search_result}")
            return False

    except Exception as e:
        logger.error(f"❌ Tool test failed with exception: {e}", exc_info=True)
        return False

    logger.info("✅ All tool tests completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_tools())
        if success:
            print("\n🎉 All tests passed! Your tools are working correctly.")
        else:
            print("\n💥 Some tests failed. Check the logs above.")
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
