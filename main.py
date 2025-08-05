import asyncio
import uuid
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from src.models.user_model import UserModel
from src.services.cache_service import RedisService
from src.services.api_service import DriversAPIClient
from src.lngraph.tools.driver_tools import DriverTools
from src.lngraph.graph import create_agent_graph
import logging
from src.models.agent_state_model import AgentState

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """
    Main function to initialize services, build the graph, and run the CLI chat loop.
    """
    # --- 1. Initialize Dependencies ---
    logger.info("Initializing services...")

    # Initialize the Language Model
    llm = ChatVertexAI(model="gemini-2.0-flash", temperature=0.5)

    # Initialize Redis Service for caching
    redis_service = RedisService()
    if not await redis_service.ping():
        logger.error("Could not connect to Redis. Please ensure it is running.")
        return

    # Initialize the API Client and Tools
    session_id = str(uuid.uuid4())
    api_client = DriversAPIClient(session_id=session_id, redis_service=redis_service)
    driver_tools = DriverTools(api_client=api_client)

    # --- 2. Build the Agent Graph ---
    logger.info("Building the agent graph...")
    app = create_agent_graph(llm, driver_tools)

    # --- 3. Run the CLI Chat Loop ---
    print("\n🚗 Cab Booking Agent is ready! Type 'exit' to end the conversation.")
    print("=" * 60)

    # CRITICAL FIX: Initialize with complete state structure including new fields
    conversation_state: AgentState = {
        "session_id": session_id,
        "messages": [],
        "user": UserModel(
            id="user123",
            username="cab_user",
            name="Cab User",
            phone_no="1234567890",
            preferred_languages=["english"],
            profile_image="default.jpg"
        ),
        # Pre-initialize ALL required state fields to prevent reset
        "last_user_message": "",
        "conversation_language": "en",
        "intent": None,
        "search_city": None,
        "current_page": 1,
        "page_size": 10,
        "radius": 100,
        "search_strategy": "hybrid",
        "use_cache": True,
        "active_filters": {},
        "previous_filters": [],
        "is_filtered": False,  # NEW field
        "total_filtered_results": 0,  # NEW field
        "current_drivers": [],
        "all_drivers": [],  # NEW field to preserve original search results
        "total_results": 0,
        "has_more_results": False,
        "selected_driver": None,
        "driver_summary": None,
        "booking_status": "none",
        "booking_details": None,
        "dropLocation": None,
        "pickupLocation": None,
        "trip_type": "one-way",
        "trip_duration": None,
        "full_trip_details": False,
        "trip_doc_id": "",
        "last_error": None,
        "retry_count": 0,
        "failed_node": None,
        "next_node": None,
        "filter_relaxation_suggestions": None,
    }

    print("💬 You can ask me to:")
    print("  • Find drivers: 'book me a cab from delhi to jaipur'")
    print("  • Get driver info: 'tell me about ramesh' or 'ramesh phone number'")
    print("  • Apply filters: 'show me SUV drivers' or 'drivers with 5+ years experience'")
    print("  • Book a ride: 'book with ramesh'")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("🚗 Agent: Thank you for using our cab booking service! Have a great day!")
                break

            if not user_input:
                print("🚗 Agent: Please tell me what you'd like to do.")
                continue

            # Append the new user message to the persistent state's history
            conversation_state["messages"].append(HumanMessage(content=user_input))

            # Clear previous errors when starting new interaction
            conversation_state["last_error"] = None

            print("🚗 Agent: ", end="", flush=True)

            # CRITICAL: Invoke the graph with the COMPLETE conversation_state
            final_state = await app.ainvoke(conversation_state)

            # The final response is the last message added by the agent
            if final_state["messages"] and len(final_state["messages"]) > 0:
                response_message = final_state["messages"][-1]
                print(f"{response_message.content}")
            else:
                print("I'm processing your request...")

            # CRITICAL: Update our local state with the final state from the graph.
            # This ensures the agent remembers the conversation for the next turn.
            conversation_state = final_state

            # Debug info (can be removed in production)
            if conversation_state.get("search_city"):
                drivers_count = len(conversation_state.get("current_drivers", []))
                all_drivers_count = len(conversation_state.get("all_drivers", []))
                filter_status = " (filtered)" if conversation_state.get("is_filtered", False) else ""
                print(f"\n[Debug: {drivers_count}/{all_drivers_count} drivers available in {conversation_state['search_city']}{filter_status}]")

        except KeyboardInterrupt:
            print("\n\n🚗 Agent: Goodbye! Hope to help you again soon!")
            break
        except Exception as e:
            logger.critical(f"An unhandled error occurred in the graph execution: {e}", exc_info=True)
            print("\n🚗 Agent: I'm sorry, I encountered a technical issue. Let me try to help you again.")
            # Reset error state but keep the conversation context
            conversation_state["last_error"] = None
            conversation_state["failed_node"] = None

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
