from langgraph.graph import StateGraph, END
from src.services.api_service import DriversAPIClient
from src.models.agent_state_model import AgentState
from src.lngraph.nodes.initialize_agent_node import InitializeAgentNode
from src.lngraph.nodes.classify_intent_node import ClassifyIntentNode
from src.lngraph.nodes.search_drivers_node import SearchDriversNode
from src.lngraph.nodes.driver_info_intent_node import GetDriverInfoNode
from src.lngraph.nodes.filter_drivers_node import FilterDriversNode
from src.lngraph.nodes.book_driver_node import BookDriverNode
from src.lngraph.nodes.response_generator_node import ResponseGeneratorNode
from src.lngraph.nodes.trip_info_collection_node import TripInfoCollectionNode
from src.lngraph.nodes.more_drivers_node import MoreDriversNode
from langchain_google_vertexai import ChatVertexAI
from src.lngraph.tools.driver_tools import DriverTools
import logging

logger = logging.getLogger(__name__)
def route_after_intent_classification(state: AgentState):
    """
    FIXED: Enhanced router with better validation and trip info prioritization.
    This router now acts as a strict gatekeeper, ensuring trip details are
    collected before allowing any search, booking, or filtering operations.
    """
    intent = state.get("intent")
    has_complete_trip_info = state.get("full_trip_details", False)
    all_drivers = state.get("all_drivers", [])

    print(f"DEBUG: Routing intent '{intent}' with has_complete_trip_info: {has_complete_trip_info}")

    # --- Primary Gatekeeper ---
    # For any intent that requires driver context, first ensure we have trip details.
    if intent in ["driver_search_intent", "booking_or_confirmation_intent", "filter_intent", "more_drivers_intent"]:
        if not has_complete_trip_info:
            return "collect_trip_info"

    # --- Intent-Based Routing (Post-Gatekeeper) ---
    if intent == "general_intent":
        return "generate_response"

    if intent == "driver_search_intent":
        return "search_drivers"

    if intent == "booking_or_confirmation_intent":
        if not all_drivers: # Can't book if no drivers have ever been searched for
            return "generate_response"
        return "book_driver"

    if intent == "driver_info_intent":
        if not all_drivers: # Can't get info if no drivers are in context
            return "generate_response"
        return "get_driver_info"

    if intent == "filter_intent":
        return "filter_drivers"

    if intent == "more_drivers_intent":
        if not all_drivers:
            return "generate_response"
        return "more_drivers"

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

def should_continue_conversation(state: AgentState):
    """
    Determines if the conversation should continue or end.
    FIXED: This function now always ends the graph turn after a response is generated.
    The main application loop is responsible for continuing the conversation.
    """
    logger.info("Graph execution complete for this turn.")
    return "end_conversation"


def create_agent_graph(llm: ChatVertexAI, driver_tools: DriverTools, api_client: DriversAPIClient):
    """
    Builds and compiles the LangGraph for the cab booking agent.
    """
    initialize_agent_node = InitializeAgentNode()
    classify_intent_node = ClassifyIntentNode(llm)
    trip_info_collection_node = TripInfoCollectionNode(llm)
    search_drivers_node = SearchDriversNode(llm, driver_tools)
    get_driver_info_node = GetDriverInfoNode(llm, driver_tools)
    filter_drivers_node = FilterDriversNode(llm, driver_tools)
    book_driver_node = BookDriverNode(llm, driver_tools)
    response_generator_node = ResponseGeneratorNode(llm, api_client)
    more_drivers_node = MoreDriversNode(driver_tools)

    workflow = StateGraph(AgentState)

    workflow.add_node("initialize_agent", initialize_agent_node.execute)
    workflow.add_node("classify_intent", classify_intent_node.execute)
    workflow.add_node("collect_trip_info", trip_info_collection_node.execute)
    workflow.add_node("search_drivers", search_drivers_node.execute)
    workflow.add_node("get_driver_info", get_driver_info_node.execute)
    workflow.add_node("filter_drivers", filter_drivers_node.execute)
    workflow.add_node("book_driver", book_driver_node.execute)
    workflow.add_node("generate_response", response_generator_node.execute)
    workflow.add_node("more_drivers", more_drivers_node.execute)

    workflow.set_entry_point("initialize_agent")
    workflow.add_edge("initialize_agent", "classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent_classification,
        {
            "collect_trip_info": "collect_trip_info",
            "search_drivers": "search_drivers",
            "get_driver_info": "get_driver_info",
            "filter_drivers": "filter_drivers",
            "book_driver": "book_driver",
            "more_drivers": "more_drivers",
            "generate_response": "generate_response"
        }
    )

    workflow.add_conditional_edges(
        "collect_trip_info",
        route_after_trip_collection,
        {
            "search_drivers": "search_drivers",
            "generate_response": "generate_response"
        }
    )

    # All action nodes lead to the response generator to inform the user
    workflow.add_edge("search_drivers", "generate_response")
    workflow.add_edge("get_driver_info", "generate_response")
    workflow.add_edge("filter_drivers", "generate_response")
    workflow.add_edge("book_driver", "generate_response")
    workflow.add_edge("more_drivers", "generate_response")

    # The response generator is the final step before the graph finishes its run.
    workflow.add_conditional_edges(
        "generate_response",
        should_continue_conversation,
        {
            "end_conversation": END,
        }
    )

    app = workflow.compile()
    return app
