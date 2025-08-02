"""
Graph structure for the cab booking agent.
Defines the conversation flow and node connections.
"""

import logging
from typing import Dict, Any, List, Literal, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

from src.models.agent_state_model import AgentState
from src.lngraph.nodes.entry_node import EntryNode
from src.lngraph.nodes.city_clarification_node import CityClarificationNode
from src.lngraph.nodes.driver_search_node import DriverSearchNode
from src.lngraph.nodes.filter_application_node import FilterApplicationNode
from src.lngraph.nodes.filter_relaxation_node import FilterRelaxationNode
from src.lngraph.nodes.driver_details_node import DriverDetailsNode
from src.lngraph.nodes.booking_confirmation_node import BookingConfirmationNode
from src.lngraph.nodes.pagination_node import PaginationNode
from src.lngraph.nodes.general_response_node import GeneralResponseNode
from src.lngraph.nodes.no_results_handler_node import NoResultsHandlerNode
from src.lngraph.nodes.error_handler_node import ErrorHandlerNode
from src.lngraph.nodes.error_recovery_node import ErrorRecoveryNode
from src.lngraph.nodes.suggest_nearby_cities_node import SuggestNearbyCitiesNode
from src.lngraph.nodes.wait_for_user_input_node import WaitForUserInputNode
from src.lngraph.nodes.manual_assistance_node import ManualAssistanceNode
from src.lngraph.nodes.booking_complete_node import BookingCompleteNode

logger = logging.getLogger(__name__)


class CabBookingGraph:
    """
    Main graph structure for the cab booking agent.
    Manages the conversation flow and node execution.
    """

    def __init__(self, llm, driver_tools):
        """
        Initialize the graph with required components.

        Args:
            llm: Language model for NLU and generation
            driver_tools: Tools for driver operations
        """
        self.llm = llm
        self.driver_tools = driver_tools
        self.graph = None
        self.compiled_graph = None

        # Initialize all nodes
        self.nodes = {
            "entry_node": EntryNode(llm),
            "city_clarification_node": CityClarificationNode(llm),
            "driver_search_node": DriverSearchNode(llm, driver_tools),
            "filter_application_node": FilterApplicationNode(llm, driver_tools),
            "filter_relaxation_node": FilterRelaxationNode(llm, driver_tools),
            "driver_details_node": DriverDetailsNode(llm, driver_tools),
            "booking_confirmation_node": BookingConfirmationNode(llm, driver_tools),
            "pagination_node": PaginationNode(llm, driver_tools),
            "general_response_node": GeneralResponseNode(llm),
            "no_results_handler_node": NoResultsHandlerNode(llm, driver_tools),
            "error_handler_node": ErrorHandlerNode(llm),
            "error_recovery_node": ErrorRecoveryNode(llm),
            "suggest_nearby_cities_node": SuggestNearbyCitiesNode(llm, driver_tools),
            "wait_for_user_input_node": WaitForUserInputNode(),
            "manual_assistance_node": ManualAssistanceNode(llm),
            "booking_complete_node": BookingCompleteNode(llm),
        }

        # Build the graph
        self._build_graph()

    def _build_graph(self):
        """Build the state graph with all nodes and edges."""
        # Create the graph
        self.graph = StateGraph(AgentState)

        # Add all nodes to the graph
        for node_name, node_instance in self.nodes.items():
            self.graph.add_node(node_name, node_instance)

        # Define the routing function
        def route_next_node(state: AgentState) -> str:
            """Route to the next node based on state."""
            next_node = state.get("next_node", "wait_for_user_input")

            # Log routing decision
            logger.debug(f"Routing from current node to: {next_node}")

            # Special handling for wait_for_user_input
            if (
                next_node == "wait_for_user_input"
                or next_node == "wait_for_user_input_node"
            ):
                # Check if we should end the conversation
                if state.get("conversation_state") == "ended":
                    return END
                # For wait state, we actually end this execution
                # The next user input will restart from entry_node
                return END

            # Validate node exists
            if next_node not in self.nodes:
                logger.warning(
                    f"Unknown next_node: {
                        next_node
                    }, defaulting to general_response_node"
                )
                return "general_response_node"

            return next_node

        # Set entry point
        self.graph.set_entry_point("entry_node")

        # Add conditional edges from each node
        for node_name in self.nodes.keys():
            if node_name != "wait_for_user_input_node":
                self.graph.add_conditional_edges(
                    node_name,
                    route_next_node,
                    # Map possible next nodes
                    {**{n: n for n in self.nodes.keys()}, END: END},
                )

        # Special handling for wait_for_user_input_node
        self.graph.add_edge("wait_for_user_input_node", END)

        # Compile the graph
        self.compiled_graph = self.graph.compile()

        logger.info("Graph built and compiled successfully")

    async def process_message(
        self, message: str, state: AgentState, session_id: str
    ) -> AgentState:
        """
        Process a user message through the graph.

        Args:
            message: User message
            state: Current conversation state
            session_id: Session identifier

        Returns:
            Updated state after processing
        """
        try:
            # Add user message to state
            from langchain_core.messages import HumanMessage, AIMessage

            if "messages" not in state:
                state["messages"] = []

            state["messages"].append(HumanMessage(content=message))
            state["session_id"] = session_id
            state["last_user_message"] = message  # Make sure this is set

            # Reset flow control
            state["awaiting_user_input"] = False

            # Log conversation context
            logger.info(
                f"Processing message for session {session_id}: {message[:50]}..."
            )

            # Run the graph starting from entry node
            config = {"recursion_limit": 10}
            result = await self.compiled_graph.ainvoke(state, config)

            # Ensure we have proper state structure
            if not isinstance(result, dict):
                result = {"messages": result} if isinstance(result, list) else state

            # Make sure we have at least one AI response
            has_ai_response = False
            for msg in reversed(result.get("messages", [])):
                # Not the welcome message
                if isinstance(msg, AIMessage) and msg != state["messages"][0]:
                    has_ai_response = True
                    break

            if not has_ai_response:
                # If no AI response was generated, add an error message
                result["messages"].append(
                    AIMessage(
                        content="I'm having trouble processing your request. Could you please try again?"
                    )
                )

            # Set awaiting input flag
            result["awaiting_user_input"] = True

            return result

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)

            # Create error response
            error_state = state.copy()
            error_state["last_error"] = str(e)
            error_state["messages"].append(
                AIMessage(
                    content="I encountered an error processing your message. Please try again."
                )
            )
            error_state["awaiting_user_input"] = True

            return error_state

    def get_graph_visualization(self) -> str:
        """
        Get a text representation of the graph structure.

        Returns:
            Graph structure as string
        """
        try:
            # Get graph structure
            nodes = list(self.nodes.keys())

            visualization = "Cab Booking Agent Graph Structure:\n"
            visualization += "=" * 50 + "\n\n"

            visualization += "Nodes:\n"
            for node in nodes:
                visualization += f"  - {node}\n"

            visualization += "\nEntry Point: entry_node\n"
            visualization += "\nKey Flows:\n"
            visualization += "  1. Search Flow: entry → search → results/no_results\n"
            visualization += "  2. Filter Flow: entry → filter → results/relaxation\n"
            visualization += "  3. Booking Flow: entry → details → booking → complete\n"
            visualization += "  4. Error Flow: any → error_handler → recovery/manual\n"

            return visualization

        except Exception as e:
            return f"Error generating visualization: {str(e)}"

    def get_node_info(self, node_name: str) -> Dict[str, Any]:
        """
        Get information about a specific node.

        Args:
            node_name: Name of the node

        Returns:
            Node information
        """
        if node_name in self.nodes:
            node = self.nodes[node_name]
            return {
                "name": node_name,
                "type": type(node).__name__,
                "description": node.__doc__.strip()
                if node.__doc__
                else "No description",
                "has_llm": hasattr(node, "llm"),
                "has_tools": hasattr(node, "driver_tools"),
            }
        return {"error": f"Node {node_name} not found"}

    async def handle_special_commands(
        self, message: str, state: AgentState
    ) -> Optional[AgentState]:
        """
        Handle special commands like /reset, /help, etc.

        Args:
            message: User message
            state: Current state

        Returns:
            Updated state or None if not a command
        """
        if not message.startswith("/"):
            return None

        command = message.lower().strip()

        if command == "/reset":
            # Reset conversation
            new_state = {
                "messages": [],
                "session_id": state.get("session_id"),
                "user_id": state.get("user_id"),
                "conversation_language": "english",
                "current_page": 1,
                "page_size": 10,
                "awaiting_user_input": True,
            }

            from langchain_core.messages import AIMessage

            new_state["messages"].append(
                AIMessage(
                    content="Conversation reset. How can I help you find a driver?"
                )
            )
            return new_state

        elif command == "/help":
            from langchain_core.messages import AIMessage

            help_message = """Available commands:
• /reset - Start a new conversation
• /help - Show this help message
• /status - Show current conversation status

Regular usage:
• Tell me your city to find drivers
• Apply filters like "show SUV drivers"
• Select and book drivers by name or number"""

            state["messages"].append(AIMessage(content=help_message))
            return state

        elif command == "/status":
            status_info = f"""Current Status:
• Session: {state.get("session_id", "Unknown")}
• City: {state.get("pickup_city", "Not set")}
• Drivers shown: {len(state.get("current_drivers", []))}
• Active filters: {len(state.get("active_filters", {}))}
• Selected driver: {state.get("selected_driver", {}).get("name", "None")}"""

            from langchain_core.messages import AIMessage

            state["messages"].append(AIMessage(content=status_info))
            return state

        return None
