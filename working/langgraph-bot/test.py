# test.py
"""Simple test to verify the agent works"""

import asyncio
from langchain_core.messages import HumanMessage

from langgraph_agent.graph.builder import app as cab_agent


def test_agent():
    """Test the agent with sample queries"""
    print("🧪 Testing Cab Booking Agent...\n")

    # Test 1: Simple city query
    print("Test 1: City query")
    state = {
        "chat_history": [HumanMessage(content="I need drivers in Jaipur")],
        "all_fetched_drivers": [],
        "drivers_with_full_details": [],
        "filtered_drivers": [],
        "applied_filters": {},
        "pickup_location": None,
        "last_bot_response": None,
        "tool_calls": [],
        "current_display_index": 0,
        "current_page": 1,
        "fetch_count": 0,
    }

    result = cab_agent.invoke(state)
    print(f"Response: {result.get('last_bot_response', 'No response')}")
    print(f"Total drivers fetched: {len(result.get('all_fetched_drivers', []))}")
    print(f"Drivers shown: 5 (pagination enabled)\n")

    # Test 2: Filter query with specific criteria
    if result.get("all_fetched_drivers"):
        print("Test 2: Filter query - SUV and age < 30")
        # Continue with the same state
        result["chat_history"].append(
            HumanMessage(content="Show me drivers with suv and age less than 30")
        )

        result = cab_agent.invoke(result)
        print(f"Response: {result.get('last_bot_response', 'No response')}")
        print(f"Filtered drivers: {len(result.get('filtered_drivers', []))}\n")

        # Test 3: Clear filter and try language filter on existing drivers
        print("Test 3: Language filter - English speaking")
        # Use the state from city fetch to ensure we have drivers
        city_state = result.copy()
        city_state["chat_history"].append(
            HumanMessage(content="show me english speaking drivers")
        )

        result = cab_agent.invoke(city_state)
        print(f"Response: {result.get('last_bot_response', 'No response')}")
        print(f"Filtered drivers: {len(result.get('filtered_drivers', []))}\n")

        # Test 4: Show more functionality - use state that has drivers
        print("Test 4: Show more drivers")
        # Go back to original city fetch state
        show_more_state = result.copy()
        show_more_state["chat_history"].append(HumanMessage(content="show more"))

        result = cab_agent.invoke(show_more_state)
        print(f"Response: {result.get('last_bot_response', 'No response')}")
        print(f"Current display index: {result.get('current_display_index', 0)}\n")

        # Test 5: Pet friendly filter
        print("Test 5: Pet friendly filter")
        pet_state = result.copy()
        pet_state["chat_history"].append(
            HumanMessage(content="show me pet friendly drivers")
        )

        result = cab_agent.invoke(pet_state)
        print(f"Response: {result.get('last_bot_response', 'No response')}")
        print(f"Filtered drivers: {len(result.get('filtered_drivers', []))}\n")

        # Test 6: Multiple filters at once
        print("Test 6: Multiple filters - sedan, married, Hindi speaking")
        multi_filter_state = city_state.copy()
        multi_filter_state["chat_history"].append(
            HumanMessage(content="show me married hindi speaking drivers with sedan")
        )

        result = cab_agent.invoke(multi_filter_state)
        print(f"Response: {result.get('last_bot_response', 'No response')}")
        print(f"Filtered drivers: {len(result.get('filtered_drivers', []))}\n")

    print("✅ Testing complete!")


def test_specific_scenarios():
    """Test specific edge cases"""
    print("\n🧪 Testing Specific Scenarios...\n")

    # Test Hindi input
    print("Test Hindi: Hindi input")
    state = {
        "chat_history": [HumanMessage(content="मुझे जयपुर में ड्राइवर चाहिए")],
        "all_fetched_drivers": [],
        "drivers_with_full_details": [],
        "filtered_drivers": [],
        "applied_filters": {},
        "pickup_location": None,
        "last_bot_response": None,
        "tool_calls": [],
        "current_display_index": 0,
        "current_page": 1,
        "fetch_count": 0,
    }

    result = cab_agent.invoke(state)
    print(
        f"Response (should be in Hindi): {
            result.get('last_bot_response', 'No response')[:100]
        }..."
    )
    print(f"Drivers fetched: {len(result.get('all_fetched_drivers', []))}\n")

    # Test direct city name
    print("Test Direct City: Just city name")
    state = {
        "chat_history": [HumanMessage(content="Delhi")],
        "all_fetched_drivers": [],
        "drivers_with_full_details": [],
        "filtered_drivers": [],
        "applied_filters": {},
        "pickup_location": None,
        "last_bot_response": None,
        "tool_calls": [],
        "current_display_index": 0,
        "current_page": 1,
        "fetch_count": 0,
    }

    result = cab_agent.invoke(state)
    print(
        f"Response (should fetch drivers): {
            result.get('last_bot_response', 'No response')[:100]
        }..."
    )
    print(f"Pickup location set: {result.get('pickup_location')}\n")


if __name__ == "__main__":
    test_agent()
    test_specific_scenarios()
