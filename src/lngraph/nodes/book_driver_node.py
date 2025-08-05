from typing import Dict, Any, Optional
from src.models.drivers_model import APIResponse
from src.models.agent_state_model import AgentState
import logging
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.lngraph.tools.driver_tools import DriverTools

logger = logging.getLogger(__name__)

# --- Pydantic Model for Structured LLM Output ---

class DriverIdentifier(BaseModel):
    """
    Pydantic model to extract how the user is identifying a driver for booking.
    """
    driver_name: Optional[str] = Field(
        description="The name of the driver the user wants to book, e.g., 'Ramesh'."
    )
    driver_index: Optional[int] = Field(
        description="The 1-based index of the driver in the list, e.g., 'book the first one' -> 1."
    )

class BookDriverNode:
    """
    Node to handle booking or confirmation intents for a specific driver.
    """

    def __init__(self, llm: ChatVertexAI, driver_tools: DriverTools):
        """
        Initializes the BookDriverNode.

        Args:
            llm: An instance of a language model for entity extraction.
            driver_tools: An instance of the DriverTools class.
        """
        self.llm = llm
        self.driver_tools = driver_tools

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes the logic to identify a driver and confirm the booking details.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary with the updated state values.
        """
        logger.info("Executing BookDriverNode...")

        user_message = state["last_user_message"]
        cache_key = self.driver_tools.api_client._generate_cache_key(
            str(state["search_city"]),
            state["current_page"]
        )

        # Get current drivers from cache
        try:
            cached_data = await self.driver_tools.api_client._get_from_cache(cache_key)
            if not cached_data:
                logger.warning("No drivers in cache to book.")
                return {
                    "last_error": "I don't have a list of drivers to choose from. Please perform a search first.",
                    "failed_node": "book_driver_node"
                }

            current_drivers = APIResponse.model_validate(cached_data).data
        except Exception as e:
            logger.error(f"Error retrieving drivers from cache: {e}")
            return {
                "last_error": "I don't have a list of drivers to choose from. Please perform a search first.",
                "failed_node": "book_driver_node"
            }

        selected_driver = state.get("selected_driver")
        target_driver_id = None

        # 1. Check if a driver is already selected in the state
        if selected_driver:
            logger.info(f"Booking with pre-selected driver: {selected_driver.name}")
            target_driver_id = selected_driver.id
        elif not current_drivers:
            logger.warning("Booking intent detected without any drivers in context.")
            return {
                "last_error": "I don't have any drivers to book. Please search for drivers first.",
                "failed_node": "book_driver_node"
            }
        else:
            # 2. If no driver is pre-selected, identify from the user's message
            logger.debug("No pre-selected driver, attempting to identify from message.")
            extract_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an entity extraction expert. From the user's message, identify the driver they want to book. They might use a name or an index (e.g., 'the first one'). Extract either the name or the 1-based index."),
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

                if identifier.driver_name:
                    for driver in current_drivers:
                        if driver.name.lower() == identifier.driver_name.lower():
                            target_driver_id = driver.id
                            break
                elif identifier.driver_index and 0 < identifier.driver_index <= len(current_drivers):
                    target_driver_id = current_drivers[identifier.driver_index - 1].id

            except Exception as e:
                logger.error(f"Error during driver identification for booking: {e}", exc_info=True)
                return {"last_error": "I'm sorry, I couldn't understand which driver you want to book.", "failed_node": "book_driver_node"}

        if not target_driver_id:
            logger.warning("Could not find a matching driver to book.")
            return {"last_error": "I couldn't find that specific driver to book. Please be more specific.", "failed_node": "book_driver_node"}

        logger.info(f"Confirming booking with driver ID: {target_driver_id}")

        # 3. Call the tool to get booking confirmation details
        try:
            tool_response = await self.driver_tools.book_or_confirm_ride_with_driver.ainvoke({
                "city": state["search_city"],
                "page": state["current_page"],
                "driverId": target_driver_id,
            })

            if tool_response.get("success"):
                logger.info(f"Successfully retrieved booking details for driver {target_driver_id}.")
                return {
                    "booking_status": "confirmed",
                    "booking_details": tool_response,
                    "last_error": None,
                }
            else:
                error_msg = tool_response.get('error', 'An unknown error occurred.')
                logger.error(f"Booking tool failed: {error_msg}")
                return {"last_error": tool_response.get("msg", error_msg), "failed_node": "book_driver_node"}
        except Exception as e:
            logger.critical(f"A critical error occurred in BookDriverNode: {e}", exc_info=True)
            return {"last_error": "A system error occurred while confirming your booking.", "failed_node": "book_driver_node"}
