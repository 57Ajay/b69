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
    This function decides which node to call next based on the classified intent.
    It acts as the main router for the agent's logic.
    """
    intent = state.get("intent")

    if intent == "driver_search_intent":
        return "search_drivers"
    elif intent == "driver_info_intent":
        return "get_driver_info"
    elif intent == "filter_intent":
        return "filter_drivers"
    elif intent == "booking_or_confirmation_intent":
        return "book_driver"
    else: # general_intent or any other case
        return "generate_response"

def create_agent_graph(llm: ChatVertexAI, driver_tools: DriverTools):
    """
    Builds and compiles the LangGraph for the cab booking agent.
    """
    # Instantiate all the nodes
    initialize_agent_node = InitializeAgentNode()
    classify_intent_node = ClassifyIntentNode(llm)
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

    # 3. Conditional routing after intent classification
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent_classification,
        {
            "search_drivers": "search_drivers",
            "get_driver_info": "get_driver_info",
            "filter_drivers": "filter_drivers",
            "book_driver": "book_driver",
            "generate_response": "generate_response"
        }
    )

    # 4. After a tool-executing node, generate a response to the user
    workflow.add_edge("search_drivers", "generate_response")
    workflow.add_edge("get_driver_info", "generate_response")
    workflow.add_edge("filter_drivers", "generate_response")
    workflow.add_edge("book_driver", "generate_response")

    # 5. After generating a response, the conversation can end for this turn.
    workflow.add_edge("generate_response", END)

    # Compile the graph into a runnable app
    app = workflow.compile()
    return app
