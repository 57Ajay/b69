from typing import Dict, Any, Literal
from src.models.agent_state_model import AgentState
import logging
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- Pydantic Model for Structured LLM Output ---

IntentType = Literal[
    "driver_search_intent",
    "driver_info_intent",
    "booking_or_confirmation_intent",
    "filter_intent",
    "more_drivers_intent",
    "general_intent",
]

class Intent(BaseModel):
    """
    Pydantic model for structuring the output of the intent classification LLM call.
    This ensures the LLM's response is predictable and easy to parse.
    """
    intent: IntentType = Field(
        description="""The classified intent of the user's message., can be
        "driver_search_intent",
        "driver_info_intent",
        "booking_or_confirmation_intent",
        "filter_intent",
        "general_intent",
        "more_drivers_intent",
        """)

class ClassifyIntentNode:
    """
    Node to classify the user's intent based on the conversation history.
    """

    def __init__(self, llm: ChatVertexAI):
        """
        Initializes the ClassifyIntentNode.

        Args:
            llm: An instance of a language model, configured for structured output.
        """
        self.llm = llm.with_structured_output(Intent)

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the intent classification logic.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary containing the classified intent and any potential errors.
        """
        logger.info("Executing ClassifyIntentNode...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at classifying user intent for a cab booking agent.
            Analyze the user's message in the context of the conversation and classify it into one of the following intents:
            - driver_search_intent: User wants to find a cab, driver, or ride. (e.g., "find me a cab in delhi", "i need a ride")
            - driver_info_intent: User is asking for more details about a specific driver already presented. (e.g., "tell me more about Ramesh", "what's his experience?")
            - booking_or_confirmation_intent: User wants to book a ride with a specific driver. (e.g., "book him for me", "confirm my ride with Suresh")
            - filter_intent: User wants to add or modify filters for an ongoing search. (e.g., "show me only SUVs", "can I find someone who speaks english?")
            - more_drivers_intent: User asks to see more drivers from the current search. (e.g., "show me more", "next page", "any other options?")
            - general_intent: User is having a general conversation, greeting, or asking something outside the scope of booking a cab. (e.g., "hello", "what's the weather like?")

            Based on the last user message, determine the most appropriate intent."""),
            ("human", "Conversation History:\n{history}\n\nUser Message: {user_message}")
        ])

        history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"][:-1]])
        user_message = state["messages"][-1].content

        chain = prompt | self.llm

        try:
            res = await chain.ainvoke({
                "history": history,
                "user_message": user_message
            })

            response = Intent.model_validate(res)

            logger.info(f"Classified intent as: {response.intent}")
            print("state city:", state["search_city"])
            return {"intent": response.intent}
        except Exception as e:
            logger.error(f"Error during intent classification: {e}", exc_info=True)
            # Default to general_intent on failure to avoid breaking the flow
            return {"intent": "general_intent", "last_error": f"Intent classification failed: {e}"}
