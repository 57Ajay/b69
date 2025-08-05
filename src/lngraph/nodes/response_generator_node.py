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
    CRITICAL: Never generate fake data, only use real state information.
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
        CRITICAL: Uses only real data from state, never generates fake information.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary containing the new AI message to be added to the history.
        """
        logger.info("Executing ResponseGeneratorNode...")

        # CRITICAL: Build response based on REAL state data only
        current_drivers = state.get("current_drivers", [])
        search_city = state.get("search_city")
        last_error = state.get("last_error")
        intent = state.get("intent")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful and friendly cab booking assistant.
            CRITICAL RULES:
            1. NEVER generate fake driver names, data, or information
            2. ONLY use the exact driver information provided in the state
            3. If no drivers are available, ask the user to search for drivers first
            4. If missing trip information, ask for pickup/drop locations
            5. Always be accurate and never hallucinate data

            Current State Information:
            - Search City: {search_city}
            - Number of Real Drivers: {driver_count}
            - Intent: {intent}
            - Last Error: {last_error}
            - Real Driver Data: {real_drivers}

            Response Guidelines:
            - If there's an error, explain it clearly and ask for the missing information
            - If there are real drivers, list them with their actual names and details
            - If no drivers but user wants info/filter, ask them to search first
            - Be helpful but never make up information
            """),
            ("human", "Conversation History:\n{history}\n\nPlease respond based on the current state.")
        ])

        chain = prompt | self.llm

        history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])

        # Prepare real driver information
        real_drivers_info = []
        for driver in current_drivers:
            vehicle_types = [v.vehicle_type for v in driver.verified_vehicles] if driver.verified_vehicles else ["unknown"]
            real_drivers_info.append(f"{driver.name} ({driver.experience} yrs exp, {', '.join(vehicle_types)})")

        try:
            response = await chain.ainvoke({
                "history": history,
                "search_city": search_city or "Not specified",
                "driver_count": len(current_drivers),
                "intent": intent or "unknown",
                "last_error": last_error or "None",
                "real_drivers": real_drivers_info if real_drivers_info else "No drivers available"
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
