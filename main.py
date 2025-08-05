import asyncio
import uuid
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from src.models.agent_state_model import AgentState
from src.models.user_model import UserModel
from src.services.cache_service import RedisService
from src.services.api_service import DriversAPIClient
from src.lngraph.tools.driver_tools import DriverTools
from src.lngraph.graph import create_agent_graph
import logging

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
    # Make sure you have authenticated with `gcloud auth application-default login`
    llm = ChatVertexAI(model="gemini-2.0-flash", temperature=0.5)

    # Initialize Redis Service for caching
    # This will connect to localhost:6379 by default, assuming you have Redis running in Docker
    redis_service = RedisService()
    if not await redis_service.ping():
        logger.error("Could not connect to Redis. Please ensure it is running.")
        return

    # Initialize the API Client and Tools
    session_id = str(uuid.uuid4()) # Generate a unique session ID for this run
    api_client = DriversAPIClient(session_id=session_id, redis_service=redis_service)
    driver_tools = DriverTools(api_client=api_client)

    # --- 2. Build the Agent Graph ---
    logger.info("Building the agent graph...")
    app = create_agent_graph(llm, driver_tools)

    # --- 3. Run the CLI Chat Loop ---
    print("\nCab Booking Agent is ready. Type 'exit' to end the conversation.")
    print("-" * 50)

    # Maintain the list of messages for the conversation history
    messages = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Agent: Goodbye!")
            break

        # Append the new user message to the history
        messages.append(HumanMessage(content=user_input))

        # The input to the graph is the current state
        # graph_input = {
        #     "session_id": session_id,
        #     "messages": messages,
        # }

        user: UserModel = UserModel(
            id="6969",
            username="69_love",
            name="mr.69",
            phone_no="69696969",
            profile_image="hello@69.com",
            preferred_languages=["english"]
        )

        graph_input: AgentState = {
            "conversation_language": "english",
            "intent": None,
            "last_user_message": user_input,
            "messages": [],
            "user": user,
            "search_city": None,
            "current_page": 1,
            "page_size": 10,
            "radius": 100,
            "search_strategy": "hybrid",
            "use_cache": True,
            "active_filters": {},
            "previous_filters": [],
            "current_drivers": [],
            "total_results": 0,
            "has_more_results": False,
            "selected_driver": None,
            "booking_status": "none",
            "booking_details": None,
            "dropLocation": None,
            "pickupLocation": None,
            "trip_type": "",
            "trip_duration": None,
            "full_trip_details": False,
            "trip_doc_id": "",
            "last_error": None,
            "retry_count": 0,
            "failed_node": None,
            "next_node": None,
            "filter_relaxation_suggestions": None,
            "session_id": session_id
        }

        try:
            # Invoke the graph
            final_state = await app.ainvoke(graph_input)

            # The final response is the last message added by the agent
            response_message = final_state["messages"][-1]
            print(f"Agent: {response_message.content}")

            # Update the message history with the agent's response
            messages = final_state["messages"]

        except Exception as e:
            logger.critical(f"An unhandled error occurred in the graph execution: {e}", exc_info=True)
            print("Agent: I'm sorry, a critical error occurred. Please try restarting the conversation.")
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
