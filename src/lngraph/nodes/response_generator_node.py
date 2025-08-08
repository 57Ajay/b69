from typing import Dict, Any
from src.models.agent_state_model import AgentState
import logging
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from src.services.api_service import DriversAPIClient

logger = logging.getLogger(__name__)


class ResponseGeneratorNode:
    """
    FIXED: Node to generate a final, user-facing response based on the agent's state.
    This version uses a powerful LLM prompt to generate natural, context-aware responses
    instead of relying on rigid, hardcoded logic.
    """

    def __init__(self, llm: ChatVertexAI, client: DriversAPIClient):
        """
        Initializes the ResponseGeneratorNode.

        Args:
            llm: An instance of a language model.
            client: API client for driver details
        """
        self.llm = llm
        self.client = client

    def _format_state_for_prompt(self, state: AgentState) -> str:
        """
        Creates a structured, human-readable string from the state for the LLM prompt.
        """
        prompt_parts = []

        # Conversation History
        history = "\n".join([f"{msg.type}: {msg.content}" for msg in state.get("messages", [])[-10:]])
        prompt_parts.append(f"### Conversation History (last 10 messages):\n{history}\n")

        # Last User Message
        prompt_parts.append(f"### Last User Message:\n{state.get('last_user_message', 'N/A')}\n")

        # Handle errors first
        if state.get("last_error"):
            prompt_parts.append(f"### Priority Task: Address this error!\nError: {state.get('last_error')}\n")
            return "".join(prompt_parts) # Stop here if there's an error

        # Booking Status
        if state.get("booking_status") == "confirmed" and state.get("booking_details"):
            details = state.get("booking_details", {})
            if details is None:
                prompt_parts.append("### Priority Task: Fetch Booking Details!\nBooking details are not available.\n")
                return "".join(prompt_parts)
            prompt_parts.append(f"### Priority Task: Confirm Booking!\nBooking is confirmed for: {details.get('Driver Name')}\nContact: {details.get('PhoneNo.')}\nProfile: {details.get('Profile')}\n")

        # Specific Driver Info
        elif state.get("driver_summary"):
            summary = state.get('driver_summary', {})
            if summary is None:
                prompt_parts.append("### Priority Task: Fetch Driver Info!\nDriver Summary is not available.\n")
                return "".join(prompt_parts)

            vehicles_str = ", ".join(summary.get("vehicles", ["Not available"]))
            languages_str = ", ".join(summary.get("languages", ["Not specified"]))
            costs_str = ", ".join([f"₹{c}" for c in summary.get("per_km_cost", [])])

            summary_text = (
                f"- Name: {summary.get('name', 'N/A')}\n"
                f"- Age: {summary.get('age', 'N/A')}\n"
                f"- Gender: {summary.get('gender', 'N/A')}\n"
                f"- Experience: {summary.get('experience', 0)} years\n"
                f"- Languages: {languages_str}\n"
                f"- Pet-Friendly: {'Yes' if summary.get('pet_allowed') else 'No'}\n"
                f"- Marital Status: {'Married' if summary.get('married') else 'Single'}\n"
                f"- Vehicles & Cost: {vehicles_str} (Cost per KM: {costs_str})\n"
                f"- Phone: {summary.get('phone', 'Not available')}\n"
                f"- Profile URL: {summary.get('profile_url', 'Not available')}"
            )
            prompt_parts.append(f"### Task: Respond to query about a specific driver\nHere is the driver's data:\n{summary_text}\n")

        # List of Drivers
        elif state.get("current_drivers"):
            drivers = state.get("current_drivers", [])
            if drivers is None:
                prompt_parts.append("### Priority Task: Fetch Driver Info!\nNO current Drivers are available.\n")
                return "".join(prompt_parts)
            filters = state.get("active_filters", {})

            driver_list_str = "\n".join([f"{i+1}. {d['driver_name']} (ID: {d['driver_id']})" for i, d in enumerate(drivers)])

            prompt_parts.append("### Task: Present a list of drivers\n")
            prompt_parts.append(f"Found {len(drivers)} drivers in {state.get('search_city')}.\n")
            if filters:
                filter_str = ", ".join([f"{k.replace('_', ' ')}: {v}" for k,v in filters.items()])
                prompt_parts.append(f"Active filters: {filter_str}\n")
            prompt_parts.append(f"Driver List:\n{driver_list_str}\n")
            if state.get("has_more_results"):
                prompt_parts.append("There are more drivers available. Mention that the user can say 'show more'.\n")

        # Initial State or need info
        else:
            prompt_parts.append("### Task: Guide the user to start a search.\n")
            if not state.get("pickupLocation"):
                prompt_parts.append("The user needs to provide a pickup location.\n")
            elif not state.get("dropLocation"):
                prompt_parts.append("The user needs to provide a drop-off location.\n")

        return "".join(prompt_parts)


    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        FIXED: Generates a response to the user by feeding the current state to an LLM.
        """
        logger.info("Executing ResponseGeneratorNode...")

        # Prepare a structured input for the LLM
        state_summary = self._format_state_for_prompt(state)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly and professional cab booking assistant.
Your goal is to provide clear, natural, and helpful responses to the user based on the current state of the conversation.
Analyze the provided state information and conversation history to craft the best possible response.

**Response Guidelines:**

1.  **Prioritize Errors:** If there is a `Priority Task: Address this error!`, your primary goal is to communicate this error to the user in a helpful and clear way. Do not mention other information.
2.  **Confirm Bookings:** If the `Priority Task` is to confirm a booking, create a cheerful confirmation message. Include the driver's name, contact number, and profile link. Wish them a safe trip.
3.  **Answer Specific Questions:** If you have data for a specific driver (`Task: Respond to query about a specific driver`), analyze the 'Last User Message' to see if they asked for specific information (e.g., "what is his phone number?", "what car does he drive?").
    - If they asked a specific question, answer ONLY that question.
    - If the user asked a general question (e.g., "tell me more about him"), provide a concise, friendly summary of the driver's details. Do NOT just dump all the data.
4.  **Present Driver Lists:** If the `Task` is to present a list of drivers, format it nicely. Mention the city and any active filters. At the end, guide the user on what to do next (e.g., "You can ask for more details about a driver, apply more filters, or ask to book one."). If more drivers are available, tell them they can say 'show more'.
5.  **Guide the User:** If there are no drivers and no errors (`Task: Guide the user`), your job is to help the user start a search. If trip details like pickup or drop location are missing, ask for them.
6.  **Be Natural:** Do not sound like a robot. Use conversational language. Avoid technical jargon.
"""),
            ("human", "Here is the current state of our conversation. Please generate the appropriate response for me to send to the user.\n\n{state_summary}")
        ])

        response_chain = prompt | self.llm

        try:
            response_content = await response_chain.ainvoke({"state_summary": state_summary})

            ai_message = AIMessage(content=response_content.content)
            logger.info(f"Generated AI Response: {ai_message.content}")

            # Clear single-turn state variables after they have been used for a response
            new_state = {
                "messages": [ai_message],
                "last_error": None,
                "driver_summary": None,
            }
            # Reset booking status after confirmation is sent
            if state.get("booking_status") == "confirmed":
                new_state["booking_status"] = "none"
                new_state["booking_details"] = None

            return new_state

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            error_message = AIMessage(content="I'm sorry, I encountered an issue while generating a response. Could you please try again?")
            return {"messages": [error_message]}
