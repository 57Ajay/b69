from typing import Dict, Any, Optional
from src.models.agent_state_model import AgentState
import logging
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.lngraph.tools.driver_tools import DriverTools
from src.models.drivers_model import APIResponse

logger = logging.getLogger(__name__)

# --- Pydantic Model for Structured LLM Output ---

class DriverIdentifier(BaseModel):
    """
    Pydantic model to extract how the user is identifying a driver.
    """
    driver_name: Optional[str] = Field(
        description="The name of the driver the user is asking about, e.g., 'Ramesh'."
    )
    driver_index: Optional[int] = Field(
        description="The 1-based index of the driver in the list, e.g., 'the first one' -> 1, 'the third driver' -> 3."
    )

class GetDriverInfoNode:
    """
    Node to handle requests for detailed information about a specific driver.
    """

    def __init__(self, llm: ChatVertexAI, driver_tools: DriverTools):
        """
        Initializes the GetDriverInfoNode.

        Args:
            llm: An instance of a language model for entity extraction.
            driver_tools: An instance of the DriverTools class.
        """
        self.llm = llm
        self.driver_tools = driver_tools

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the logic to identify a driver and fetch their details.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary with the updated state values.
        """
        logger.info("Executing GetDriverInfoNode...")

        user_message = state["last_user_message"]

        # Check if we have search_city in state
        if not state.get("search_city"):
            logger.warning("No search city in state for driver info request.")
            return {
                "last_error": "I don't have a search location. Please search for drivers first by specifying a city.",
                "failed_node": "get_driver_info_node"
            }

        cache_key = self.driver_tools.api_client._generate_cache_key(
            str(state["search_city"]),
            state["current_page"]
        )

        # Fix: Properly handle None response from cache
        try:
            cached_data = await self.driver_tools.api_client._get_from_cache(cache_key)
            if cached_data is None:
                logger.warning("No drivers in cache to get info for.")
                return {
                    "last_error": "I don't have a list of drivers to choose from. Please perform a search first.",
                    "failed_node": "get_driver_info_node"
                }

            # Validate the cached data before accessing .data
            api_response = APIResponse.model_validate(cached_data)
            current_drivers = api_response.data

        except Exception as e:
            logger.error(f"Error retrieving drivers from cache: {e}")
            return {
                "last_error": "I don't have a list of drivers to choose from. Please perform a search first.",
                "failed_node": "get_driver_info_node"
            }

        if not current_drivers:
            logger.warning("No drivers in cached data.")
            return {
                "last_error": "I don't have a list of drivers to choose from. Please perform a search first.",
                "failed_node": "get_driver_info_node"
            }

        # 1. Extract which driver the user is asking about
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an entity extraction expert. From the user's message, identify the driver they are asking about. They might use a name or an index (e.g., 'the first one', 'the second driver'). Extract either the name or the 1-based index."),
            ("human", "Available driver names: {driver_names}\n\nUser Message: {user_message}")
        ])

        driver_names = [driver.name for driver in current_drivers]
        extract_chain = extract_prompt | self.llm.with_structured_output(DriverIdentifier)

        try:
            raw = await extract_chain.ainvoke({
                "driver_names": ", ".join(driver_names),
                "user_message": user_message
            })
            identifier = DriverIdentifier.model_validate(raw)
        except Exception as e:
            logger.error(f"Error during driver identification: {e}", exc_info=True)
            return {"last_error": "I'm sorry, I couldn't understand which driver you're asking about.", "failed_node": "get_driver_info_node"}

        # 2. Find the driver's ID from the identifier
        target_driver_id = None
        if identifier.driver_name:
            for driver in current_drivers:
                if driver.name.lower() == identifier.driver_name.lower():
                    target_driver_id = driver.id
                    break
        elif identifier.driver_index and 0 < identifier.driver_index <= len(current_drivers):
            target_driver_id = current_drivers[identifier.driver_index - 1].id

        if not target_driver_id:
            logger.warning(f"Could not find a matching driver for identifier: {identifier.model_dump_json()}")
            return {"last_error": "I couldn't find that specific driver in the current list. Please try again.", "failed_node": "get_driver_info_node"}

        logger.info(f"Fetching details for driver ID: {target_driver_id}")

        # 3. Call the tool to get driver details
        try:
            tool_response = await self.driver_tools.get_driver_info_tool.ainvoke({
                "city": state["search_city"],
                "page": state["current_page"],
                "driverId": target_driver_id,
            })

            if tool_response.get("success"):
                logger.info(f"Successfully fetched details for driver {target_driver_id}.")
                return {
                    "selected_driver": tool_response.get("driver"),
                    "last_error": None,
                }
            else:
                error_msg = tool_response.get('error', 'An unknown error occurred.')
                logger.error(f"Get driver info tool failed: {error_msg}")
                return {"last_error": tool_response.get("msg", error_msg), "failed_node": "get_driver_info_node"}
        except Exception as e:
            logger.critical(f"A critical error occurred in GetDriverInfoNode: {e}", exc_info=True)
            return {"last_error": "A system error occurred while fetching driver details.", "failed_node": "get_driver_info_node"}
