# langgraph_agent/graph/builder.py
"""LangGraph agent builder"""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langchain_google_vertexai import ChatVertexAI

from langgraph_agent.graph.sys_prompt import bot_prompt
from langgraph_agent.tools.drivers_tools import (
    get_drivers_for_city,
    get_driver_details,
    remove_filters_from_search,
    show_more_drivers,
    create_trip,
    check_driver_availability,
)
import config

logger = logging.getLogger(__name__)

# Tools list
tools = [
    get_drivers_for_city,
    get_driver_details,
    remove_filters_from_search,
    show_more_drivers,
    create_trip,
    check_driver_availability,
]

# Initialize LLM
llm = ChatVertexAI(model="gemini-2.0-flash", temperature=0.9)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: dict) -> dict:
    """Agent node that processes messages and decides actions"""
    logger.info("---AGENT NODE---")

    # Get the current date to provide context to the LLM
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    prompt_with_date = bot_prompt.format(current_date=current_date_str)


    # Build messages
    messages = [SystemMessage(content=prompt_with_date)]

    # Add chat history
    chat_history = state.get("chat_history", [])
    if chat_history:
        messages.extend(chat_history)

    # Get LLM response
    try:
        ai_response = llm_with_tools.invoke(messages)

        # Update chat history
        updated_history = chat_history + [ai_response]

        # Check for tool calls
        if not ai_response.tool_calls:
            # Direct response
            logger.info("Agent provided direct response")
            return {
                **state,
                "chat_history": updated_history,
                "last_bot_response": ai_response.content,
                "tool_calls": [],
            }
        else:
            # Agent wants to call tools
            logger.info(
                f"Agent calling tools: {[tc['name'] for tc in ai_response.tool_calls]}"
            )
            return {
                **state,
                "chat_history": updated_history,
                "tool_calls": ai_response.tool_calls,
            }

    except Exception as e:
        logger.error(f"Error in agent_node: {e}")
        return {
            **state,
            "last_bot_response": "I apologize, but I encountered an issue. Please try again.",
            "tool_calls": [],
        }


def tool_executor_node(state: dict) -> dict:
    """Execute tools requested by the agent, now with API-side filtering."""
    logger.info("---TOOL EXECUTOR NODE---")

    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        logger.warning("Tool executor called but no tool_calls in state.")
        return state

    tool_map = {tool.name: tool for tool in tools}
    tool_messages = []
    state_updates = dict(state)

    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id")

        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

        tool_to_call = tool_map.get(tool_name)
        if not tool_to_call:
            error_msg = f"Error: Tool '{tool_name}' not found."
            logger.error(error_msg)
            tool_messages.append(
                ToolMessage(content=error_msg, tool_call_id=tool_id, name=tool_name)
            )
            continue

        try:
            # Add context from state to tool arguments if needed
            if tool_name == "get_driver_details":
                tool_args["drivers"] = state_updates.get("all_fetched_drivers", [])

            if tool_name == "create_trip":
                tool_args["customer_details"] = {
                    "id": state_updates.get("customer_id"),
                    "name": state_updates.get("customer_name"),
                    "phone": state_updates.get("customer_phone"),
                    "profile_image": state_updates.get("customer_profile"),
                }
            if tool_name == "check_driver_availability":
                tool_args["trip_id"] = state_updates.get("trip_id")
                tool_args["pickup_location"] = state_updates.get("pickup_location")
                tool_args["drop_location"] = state_updates.get("drop_location")
                tool_args["trip_type"] = state_updates.get("trip_type")
                tool_args["customer_details"] = {
                    "id": state_updates.get("customer_id"),
                    "name": state_updates.get("customer_name"),
                    "phone": state_updates.get("customer_phone"),
                    "profile_image": state_updates.get("customer_profile"),
                }

            # ***MODIFICATION: Pass filters directly to the get_drivers_for_city tool***
            if tool_name == "get_drivers_for_city":
                # Combine new filters from the LLM with any existing filters in the state
                current_filters = state_updates.get("applied_filters", {})
                new_filters = tool_args.get("filters", {})
                current_filters.update(new_filters)
                tool_args["filters"] = current_filters
                state_updates["applied_filters"] = current_filters # Persist combined filters
                state_updates["pickup_location"] = tool_args.get("city")

            # Execute the tool
            output = tool_to_call.invoke(tool_args)

            # Update state based on the tool's output
            if tool_name == "create_trip":
                if "error" not in output:
                    state_updates["trip_id"] = output.get("tripId")
                    state_updates["pickup_location"] = tool_args.get("pickup_city")
                    state_updates["drop_location"] = tool_args.get("drop_city")
                    state_updates["trip_type"] = tool_args.get("trip_type")
                    output["message"] = f"Trip created successfully from {tool_args.get('pickup_city')} to {tool_args.get('drop_city')}. Now I will find drivers for you."
            elif tool_name == "get_drivers_for_city":
                new_drivers = output.get("drivers", [])

                # If it's a new search (page 1), reset the driver list
                if tool_args.get("page", 1) == 1:
                    state_updates["all_fetched_drivers"] = new_drivers
                    state_updates["current_display_index"] = 0
                    state_updates["fetch_count"] = 1
                else: # Append drivers for pagination
                    all_drivers = state_updates.get("all_fetched_drivers", [])
                    all_drivers.extend(new_drivers)
                    state_updates["all_fetched_drivers"] = all_drivers
                    state_updates["fetch_count"] = state_updates.get("fetch_count", 0) + 1

                # The filtered list is now the same as the fetched list
                state_updates["filtered_drivers"] = state_updates["all_fetched_drivers"]
                state_updates["current_page"] = output.get("page", 1)

                logger.info(
                    f"Fetched {len(new_drivers)} drivers, total now: {len(state_updates['all_fetched_drivers'])}"
                )

            elif tool_name == "show_more_drivers":
                info = output
                state_updates["current_display_index"] = info.get("next_index", 0)
                if info.get("should_fetch_new"):
                    current_page = state_updates.get("current_page", 1)
                    if state_updates.get("fetch_count", 0) < config.MAX_FETCH_DEPTH:
                         output = {
                            "message": "need_more_drivers",
                            "next_page": current_page + 1,
                        }

            elif tool_name == "remove_filters_from_search":
                keys_to_remove = tool_args.get("keys_to_remove", [])
                current_filters = state_updates.get("applied_filters", {}).copy()

                if "all" in keys_to_remove:
                    state_updates["applied_filters"] = {}
                else:
                    for key in keys_to_remove:
                        current_filters.pop(key, None)
                    state_updates["applied_filters"] = current_filters

                # After removing filters, we need to signal a new search
                output = {"message": "Filters removed. Please search again to see updated results."}
                state_updates["all_fetched_drivers"] = []
                state_updates["filtered_drivers"] = []
                state_updates["current_display_index"] = 0
                state_updates["fetch_count"] = 0

            # Format the output for the LLM and create a ToolMessage
            output_str = format_tool_output(tool_name, output, state_updates)
            tool_messages.append(
                ToolMessage(content=output_str, tool_call_id=tool_id, name=tool_name)
            )

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            tool_messages.append(
                ToolMessage(
                    content=f"Error: An unexpected error occurred while running the tool '{tool_name}'.",
                    tool_call_id=tool_id,
                    name=tool_name,
                )
            )

    # Update the chat history and clear the tool calls for the next cycle
    state_updates["chat_history"] = state.get("chat_history", []) + tool_messages
    state_updates["tool_calls"] = []

    return state_updates


def format_tool_output(tool_name: str, output: Any, state: dict) -> str:
    """Format tool output for LLM"""
    if tool_name == "get_drivers_for_city":
        # Get drivers to display (first 5)
        all_drivers = state.get("all_fetched_drivers", [])
        drivers_to_show = all_drivers[: config.DRIVERS_PER_DISPLAY]

        if not drivers_to_show:
            return json.dumps(
                {"total_drivers_found": 0, "message": "No drivers found."}
            )

        summary = {
            "total_drivers_fetched": len(all_drivers),
            "showing_drivers": len(drivers_to_show),
            "has_more": len(all_drivers) > config.DRIVERS_PER_DISPLAY,
            "drivers": format_drivers_list(drivers_to_show),
        }

        return json.dumps(summary, indent=2)

    elif tool_name == "show_more_drivers":
        if output.get("message") == "need_more_drivers":
            return json.dumps(
                {"status": "need_fetch_more", "next_page": output.get("next_page")}
            )

        # Get next batch to show
        current_index = state.get("current_display_index", 0)
        drivers_list = state.get("filtered_drivers", [])

        drivers_to_show = drivers_list[
            current_index : current_index + config.DRIVERS_PER_DISPLAY
        ]

        summary = {
            "showing_drivers": len(drivers_to_show),
            "has_more": current_index + config.DRIVERS_PER_DISPLAY < len(drivers_list),
            "drivers": format_drivers_list(drivers_to_show),
        }

        return json.dumps(summary, indent=2)

    elif tool_name == "get_driver_details":
        if not output:
            return json.dumps({"error": "Driver not found"})
        return json.dumps(format_drivers_list([output])[0], indent=2)


    elif isinstance(output, (dict, list)):
        return json.dumps(output, indent=2)
    else:
        return str(output)


def format_drivers_list(drivers: List[Dict]) -> List[Dict]:
    """Format driver list for display"""
    formatted = []

    for driver in drivers:
        # Get first vehicle info
        vehicle = {}
        if driver.get("vehicles") and len(driver["vehicles"]) > 0:
            first_vehicle = driver["vehicles"][0]
            vehicle = {
                "model": first_vehicle.get("model", "N/A"),
                "type": first_vehicle.get("type", "N/A"),
                "price_per_km": first_vehicle.get("per_km_cost", "N/A"),
                # Include image URL
                "image_url": first_vehicle.get("image_url"),
            }

        formatted.append(
            {
                "id": driver.get("id"),
                "name": driver.get("name"),
                "phone": driver.get("phone"),
                "username": driver.get("username"),
                "profile_image": driver.get("profile_image"),
                "age": driver.get("age"),
                "experience": driver.get("experience"),
                "languages": driver.get("languages", []),
                "is_pet_allowed": driver.get("is_pet_allowed"),
                "is_married": driver.get("is_married"),
                "city": driver.get("city"),
                "vehicle": vehicle,
                "lastAccess": driver.get("lastAccess"),
                # Include all vehicles for image requests
                "vehicles": driver.get("vehicles", []),
            }
        )

    return formatted


def route_after_agent(state: dict) -> str:
    """Router to decide next step after agent"""
    if state.get("tool_calls"):
        return "action"
    else:
        return END


# Build the graph
def create_graph():
    """Create the LangGraph workflow"""
    workflow = StateGraph(dict)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("action", tool_executor_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent", route_after_agent, {"action": "action", END: END}
    )

    # After tools, go back to agent
    workflow.add_edge("action", "agent")

    return workflow.compile()


# Create the app
app = create_graph()