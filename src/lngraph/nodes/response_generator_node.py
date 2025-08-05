from typing import Dict, Any
from src.models.agent_state_model import AgentState
import logging
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

class ResponseGeneratorNode:
    """
    Node to generate a final, user-facing response based on the agent's state.
    """

    def __init__(self, llm: ChatVertexAI):
        """
        Initializes the ResponseGeneratorNode.

        Args:
            llm: An instance of a language model.
        """
        self.llm = llm

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Generates a response to the user based on the current state.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary containing the new AI message to be added to the history.
        """
        logger.info("Executing ResponseGeneratorNode...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful and friendly cab booking assistant.
            Your goal is to provide a concise and relevant response to the user based on the current state of the conversation.

            - If there are drivers in the state, list their names, experience, and vehicle types.
            - If a driver has been selected, provide their detailed information.
            - If a booking is confirmed, state it clearly and provide the booking details.
            - If there is an error, apologize and state the error message clearly.
            - If the last intent was 'general_intent', have a friendly, general conversation.
            - If clarification is needed (e.g., for a city), ask the user for the necessary information.
            """),
            ("human", "Conversation History:\n{history}\n\nHere is the current state of our conversation, please respond to the user based on this:\n{agent_state}")
        ])

        chain = prompt | self.llm

        history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])

        # We pass a simplified version of the state to the LLM to avoid overwhelming it.
        selected_driver = state.get("selected_driver")
        relevant_state = {
            "last_user_message": state["last_user_message"],
            "intent": state["intent"],
            "search_city": state["search_city"],
            "active_filters": state["active_filters"],
            "current_drivers_summary": [f"{d.name} ({d.experience} yrs exp, {', '.join([v.vehicle_type for v in d.verified_vehicles])})" for d in state.get("current_drivers", [])],
            "selected_driver": selected_driver.model_dump(by_alias=True) if selected_driver else None,
            "booking_status": state["booking_status"],
            "booking_details": state.get("booking_details"),
            "last_error": state.get("last_error")
        }

        try:
            response = await chain.ainvoke({
                "history": history,
                "agent_state": str(relevant_state)
            })

            # The response from the LLM is the content of the AIMessage
            ai_message = AIMessage(content=response.content)

            logger.info(f"Generated AI Response: {response.content}")

            # We return a dictionary that appends this new message to the 'messages' list
            return {"messages": [ai_message]}

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            error_message = AIMessage(content="I'm sorry, I encountered an issue while generating a response. Could you please try again?")
            return {"messages": [error_message]}
