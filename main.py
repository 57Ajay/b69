
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
    # Using the model and temperature you specified.
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
    print("\nCab Booking Agent is ready. Type 'exit' to end the conversation.")
    print("-" * 50)

    # This is our persistent state for the entire conversation.
    # The graph's entry node will populate the full state on the first run.
    # We only need to manage the core inputs that change turn-by-turn.
    conversation_state = {
        "session_id": session_id,
        "messages": [],
        "user": UserModel(
            id="6969",
            username="69_love",
            name="mr.69",
            phone_no="69696969",
            preferred_languages=["english"],
            profile_image="69.com"
        )
    }

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Agent: Goodbye!")
            break

        # Append the new user message to the persistent state's history
        conversation_state["messages"].append(HumanMessage(content=user_input))

        try:
            # Invoke the graph with the UPDATED conversation_state
            final_state = await app.ainvoke(conversation_state)

            # The final response is the last message added by the agent
            response_message = final_state["messages"][-1]
            print(f"Agent: {response_message.content}")

            # CRITICAL: Update our local state with the final state from the graph.
            # This ensures the agent remembers the conversation for the next turn.
            conversation_state = final_state

        except Exception as e:
            logger.critical(f"An unhandled error occurred in the graph execution: {e}", exc_info=True)
            print("Agent: I'm sorry, a critical error occurred. Please try restarting the conversation.")
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
