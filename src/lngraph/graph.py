import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from src.models.agent_state_model import AgentState
from src.lngraph.nodes.initialize_agent_node import InitializeAgentNode
from src.lngraph.nodes.classify_intent_node import ClassifyIntentNode
from src.lngraph.nodes.search_drivers_node import SearchDriversNode
from src.lngraph.nodes.driver_info_intent_node import GetDriverInfoNode
from src.lngraph.nodes.filter_drivers_node import FilterDriversNode
from src.lngraph.nodes.book_driver_node import BookDriverNode
from src.lngraph.nodes.response_generator_node import ResponseGeneratorNode
from src.lngraph.nodes.trip_info_collection_node import TripInfoCollectionNode
from langchain_google_vertexai import ChatVertexAI
from src.lngraph.tools.driver_tools import DriverTools

# --- Graph State and Message Reducer ---

# This defines how messages are accumulated in the state.
# `operator.add` ensures that new messages are appended to the existing list.
class GraphState(TypedDict):
    state: AgentState
    messages: Annotated[list, operator.add]

def route_after_intent_classification(state: AgentState):
    """
    CRITICAL: Enhanced router with strict state validation and proper flow control.
    """
    intent = state.get("intent")
    search_city = state.get("search_city")
    current_drivers = state.get("current_drivers", [])

    print(f"DEBUG: Routing intent '{intent}' with search_city: {search_city}, drivers: {len(current_drivers)}")

    # Check if we have essential trip information for booking-related intents
    has_complete_trip_info = (
        state.get("pickupLocation") and
        state.get("dropLocation") and
        state.get("trip_type") and
        (state.get("trip_type") != "round-trip" or state.get("trip_duration"))
    )

    # For general intents - just generate a response
    if intent == "general_intent":
        return "generate_response"

    # For booking intent - STRICT validation and proper flow
    if intent == "booking_or_confirmation_intent":
        if not has_complete_trip_info:
            print("DEBUG: Missing trip info for booking, collecting trip info")
            return "collect_trip_info"
        if not search_city or not current_drivers:
            print("DEBUG: No drivers available for booking, need to search first")
            return "search_drivers"
        return "book_driver"

    # For driver search - collect trip info first if missing, then search
    if intent == "driver_search_intent":
        if not has_complete_trip_info:
            print("DEBUG: Missing trip info for search, collecting trip info")
            return "collect_trip_info"
        return "search_drivers"

    # CRITICAL: For driver info intent, MUST have existing drivers
    if intent == "driver_info_intent":
        if not search_city or not current_drivers:
            print("DEBUG: No drivers available for info request, asking to search first")
            return "generate_response"  # Will ask user to search first
        return "get_driver_info"

    # For filter intent, MUST have existing drivers
    if intent == "filter_intent":
        if not search_city or not current_drivers:
            print("DEBUG: No drivers available for filtering, asking to search first")
            return "generate_response"  # Will ask user to search first
        return "filter_drivers"

    # Default case
    return "generate_response"

def route_after_trip_collection(state: AgentState):
    """
    Router to determine next step after trip info collection.
    """
    has_complete_trip_info = state.get("full_trip_details", False)

    if has_complete_trip_info:
        print("DEBUG: Trip info complete, proceeding to search drivers")
        return "search_drivers"
    else:
        print("DEBUG: Trip info incomplete, generating response to ask for missing info")
        return "generate_response"

def create_agent_graph(llm: ChatVertexAI, driver_tools: DriverTools):
    """
    Builds and compiles the LangGraph for the cab booking agent.
    """

    # Instantiate all the nodes
    initialize_agent_node = InitializeAgentNode()
    classify_intent_node = ClassifyIntentNode(llm)
    trip_info_collection_node = TripInfoCollectionNode(llm)
    search_drivers_node = SearchDriversNode(llm, driver_tools)
    get_driver_info_node = GetDriverInfoNode(llm, driver_tools)
    filter_drivers_node = FilterDriversNode(llm, driver_tools)
    book_driver_node = BookDriverNode(llm, driver_tools)
    response_generator_node = ResponseGeneratorNode(llm)

    # Define the graph
    workflow = StateGraph(AgentState)

    # Add nodes to the graph
    workflow.add_node("initialize_agent", initialize_agent_node.execute)
    workflow.add_node("classify_intent", classify_intent_node.execute)
    workflow.add_node("collect_trip_info", trip_info_collection_node.execute)
    workflow.add_node("search_drivers", search_drivers_node.execute)
    workflow.add_node("get_driver_info", get_driver_info_node.execute)
    workflow.add_node("filter_drivers", filter_drivers_node.execute)
    workflow.add_node("book_driver", book_driver_node.execute)
    workflow.add_node("generate_response", response_generator_node.execute)

    # --- Define the graph's edges and flow ---

    # 1. Entry point
    workflow.set_entry_point("initialize_agent")

    # 2. From initialization, always classify intent
    workflow.add_edge("initialize_agent", "classify_intent")

    # 3. Enhanced conditional routing after intent classification
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent_classification,
        {
            "collect_trip_info": "collect_trip_info",
            "search_drivers": "search_drivers",
            "get_driver_info": "get_driver_info",
            "filter_drivers": "filter_drivers",
            "book_driver": "book_driver",
            "generate_response": "generate_response"
        }
    )

    # 4. After collecting trip info, check if complete
    workflow.add_conditional_edges(
        "collect_trip_info",
        route_after_trip_collection,
        {
            "search_drivers": "search_drivers",
            "generate_response": "generate_response"
        }
    )

    # 5. After tool-executing nodes, generate a response
    workflow.add_edge("search_drivers", "generate_response")
    workflow.add_edge("get_driver_info", "generate_response")
    workflow.add_edge("filter_drivers", "generate_response")
    workflow.add_edge("book_driver", "generate_response")

    # 6. After generating a response, the conversation can end for this turn.
    workflow.add_edge("generate_response", END)

    # Compile the graph into a runnable app
    app = workflow.compile()
    return app
