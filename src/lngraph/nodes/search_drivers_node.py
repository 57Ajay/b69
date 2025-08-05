from typing import Dict, Any, Optional
from src.models.agent_state_model import AgentState
import logging
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.lngraph.tools.driver_tools import DriverTools

logger = logging.getLogger(__name__)

# --- Pydantic Model for Structured LLM Output ---

class SearchEntities(BaseModel):
    """
    Pydantic model for extracting entities required for a driver search.
    """
    city: Optional[str] = Field(
        default=None,
        description="The city where the user wants to find a driver, e.g., 'delhi', 'mumbai'."
    )

class SearchDriversNode:
    """
    Node to handle the driver search intent. It extracts necessary entities,
    calls the appropriate tool, and updates the agent's state.
    """

    def __init__(self, llm: ChatVertexAI, driver_tools: DriverTools):
        """
        Initializes the SearchDriversNode.

        Args:
            llm: An instance of a language model for entity extraction.
            driver_tools: An instance of the DriverTools class.
        """
        self.llm = llm
        self.driver_tools = driver_tools

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the driver search logic.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary with the updated state values.
        """
        logger.info("Executing SearchDriversNode...")

        city = state.get("search_city")
        user_message = state["last_user_message"]

        # 1. Extract city if not already in state
        if not city:
            logger.debug("City not in state, attempting to extract from message.")
            extract_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an entity extraction expert. From the user's message, extract the city they want to search for a cab in.
                Only extract if a city is explicitly mentioned. If no city is mentioned, return null for the city field.
                Examples:
                - "find me a cab in delhi" -> city: "delhi"
                - "book me a cab" -> city: null
                - "i need a ride from mumbai" -> city: "mumbai" """),
                ("human", "{user_message}")
            ])
            extract_chain = extract_prompt | self.llm.with_structured_output(SearchEntities)
            try:
                raw = await extract_chain.ainvoke({"user_message": user_message})
                entities = SearchEntities.model_validate(raw)
                city = entities.city
            except Exception as e:
                logger.error(f"Error during entity extraction: {e}", exc_info=True)
                return {"last_error": "Failed to understand the city from your message.", "failed_node": "search_drivers_node"}

        # 2. Check if we have a city to search for
        if not city:
            logger.warning("No city found in message or state. Asking user for clarification.")
            # This will be routed to a response generator to ask the user for a city.
            return {"last_error": "I need to know which city you're looking for a cab in. Please specify the city.", "failed_node": "search_drivers_node"}

        logger.info(f"Performing driver search in city: {city}")

        # 3. Call the search tool
        try:
            tool_response = await self.driver_tools.search_drivers_tool.ainvoke({
                "city": city,
                "page": state["current_page"],
                "limit": state["page_size"],
            })

            if tool_response.get("success"):
                driver_count = tool_response.get('count', 0)
                logger.info(f"Successfully found {driver_count} drivers.")
                return {
                    "search_city": city,
                    "current_drivers": tool_response.get("drivers", []),
                    "total_results": tool_response.get("total", 0),
                    "has_more_results": tool_response.get("has_more", False),
                    "last_error": None, # Clear previous errors on success
                }
            else:
                error_msg = tool_response.get('error', 'An unknown error occurred while searching.')
                logger.error(f"Driver search tool failed: {error_msg}")
                return {
                    "last_error": tool_response.get("msg", error_msg),
                    "failed_node": "search_drivers_node",
                    "current_drivers": [], # Clear drivers on failure
                }
        except Exception as e:
            logger.critical(f"A critical error occurred in SearchDriversNode: {e}", exc_info=True)
            return {
                "last_error": "A system error occurred. Please try again later.",
                "failed_node": "search_drivers_node",
                "current_drivers": [],
            }
