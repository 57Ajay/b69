"""
Test main file for debugging the cab booking agent.
Simplified version to identify issues.
"""

import asyncio
import logging
from langchain_google_vertexai import ChatVertexAI
from src.services.api_service import DriversAPIClient
from src.lngraph.tools.driver_tools import DriverTools
from src.lngraph.nodes.entry_node import EntryNode
from src.lngraph.nodes.driver_search_node import DriverSearchNode
from src.lngraph.nodes.city_clarification_node import CityClarificationNode
from langchain_core.messages import HumanMessage, AIMessage

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def test_direct_node_execution():
    """Test nodes directly without the graph."""
    print("\n=== Testing Direct Node Execution ===\n")

    # Initialize components
    llm = ChatVertexAI(model="gemini-1.5-flash", temperature=0.7)
    api_client = DriversAPIClient("test_session", 5)
    driver_tools = DriverTools(api_client)

    # Create nodes
    entry_node = EntryNode(llm)
    search_node = DriverSearchNode(llm, driver_tools)
    city_node = CityClarificationNode(llm)

    # Test state
    state = {
        "messages": [HumanMessage(content="Show me drivers in Delhi")],
        "last_user_message": "Show me drivers in Delhi",
        "session_id": "test_session",
        "conversation_language": "english",
        "current_page": 1,
        "page_size": 10,
        "current_drivers": [],
        "active_filters": {},
    }

    # Test entry node
    print("1. Testing Entry Node...")
    try:
        state = await entry_node(state)
        print(f"Intent: {state.get('intent')}")
        print(f"Pickup City: {state.get('pickup_city')}")
        print(f"Next Node: {state.get('next_node')}")
    except Exception as e:
        print(f"Entry node error: {e}")

    # If city found, test search
    if state.get("pickup_city") and state.get("next_node") == "driver_search_node":
        print("\n2. Testing Driver Search Node...")
        try:
            state = await search_node(state)
            print(f"Drivers found: {len(state.get('current_drivers', []))}")

            # Show last message
            for msg in reversed(state.get("messages", [])):
                if isinstance(msg, AIMessage):
                    print(f"Response: {msg.content[:200]}...")
                    break
        except Exception as e:
            print(f"Search node error: {e}")

    await api_client.close()


async def test_simple_flow():
    """Test a simple conversation flow."""
    print("\n=== Testing Simple Conversation Flow ===\n")

    from src.lngraph.builder import CabBookingAgentBuilder

    # Create agent with simple config
    agent = CabBookingAgentBuilder(
        llm_model="gemini-1.5-flash", use_redis=False, enable_logging=True
    )

    # Create session
    session_id = await agent.create_session()
    print(f"Session created: {session_id}\n")

    # Test messages
    test_queries = [
        "Find drivers in Delhi",
        "Show me SUV drivers",
        "Tell me about the first driver",
    ]

    for query in test_queries:
        print(f"\nUser: {query}")
        try:
            response = await agent.process_message(session_id, query)

            if "error" in response:
                print(f"Error: {response['error']}")
            else:
                print(f"Assistant: {response['message'][:300]}...")
                if response["context"]["drivers_shown"] > 0:
                    print(f"[Found {response['context']['drivers_shown']} drivers]")
        except Exception as e:
            print(f"Error processing: {e}")
            logger.exception("Full error:")

    await agent.cleanup()


async def main():
    """Run tests."""
    # First test direct node execution
    await test_direct_node_execution()

    # Then test full flow
    await test_simple_flow()


if __name__ == "__main__":
    asyncio.run(main())
