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
        default=None,
        description="The name of the driver the user wants to book, e.g., 'Ramesh'."
    )
    driver_index: Optional[int] = Field(
        default=None,
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

    def _find_driver_by_name(self, drivers, search_name):
        """
        Find driver by name with fuzzy matching.
        """
        search_name_lower = search_name.lower()

        # First try exact match
        for driver in drivers:
            if driver.name.lower() == search_name_lower:
                return driver

        # Try partial match (driver name contains search term)
        for driver in drivers:
            if search_name_lower in driver.name.lower():
                return driver

        # Try reverse partial match (search term contains driver name parts)
        for driver in drivers:
            driver_name_parts = driver.name.lower().split()
            for part in driver_name_parts:
                if part in search_name_lower and len(part) > 2:  # Avoid matching very short words
                    return driver

        return None

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

        # CRITICAL: Use ALL drivers, not just current filtered ones
        all_drivers = state.get("all_drivers", [])
        selected_driver = state.get("selected_driver")
        target_driver = None

        # 1. Check if a driver is already selected in the state
        if selected_driver:
            logger.info(f"Booking with pre-selected driver: {selected_driver.name}")
            target_driver = selected_driver
        elif not all_drivers:
            logger.warning("Booking intent detected without any drivers in context.")
            return {
                "last_error": "I don't have any drivers to book. Please search for drivers first.",
                "failed_node": "book_driver_node"
            }
        else:
            # 2. If no driver is pre-selected, identify from the user's message
            logger.debug("No pre-selected driver, attempting to identify from message.")
            extract_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an entity extraction expert. From the user's message, identify the driver they want to book. They might use a name, pronoun like 'him/her', or an index (e.g., 'the first one'). Extract either the name or the 1-based index. Return null for fields not mentioned."),
                ("human", "Available driver names: {driver_names}\n\nUser Message: {user_message}")
            ])

            driver_names = [driver.name for driver in all_drivers]
            extract_chain = extract_prompt | self.llm.with_structured_output(DriverIdentifier)

            try:
                raw = await extract_chain.ainvoke({
                    "driver_names": ", ".join(driver_names),
                    "user_message": user_message
                })
                identifier = DriverIdentifier.model_validate(raw)

                if identifier.driver_name:
                    target_driver = self._find_driver_by_name(all_drivers, identifier.driver_name)
                elif identifier.driver_index and 0 < identifier.driver_index <= len(all_drivers):
                    target_driver = all_drivers[identifier.driver_index - 1]

                # Fallback: If user says "book with him/her" and there's a recently selected driver
                if not target_driver and selected_driver and any(word in user_message.lower() for word in ['him', 'her', 'them', 'that driver']):
                    target_driver = selected_driver

            except Exception as e:
                logger.error(f"Error during driver identification for booking: {e}", exc_info=True)
                return {"last_error": "I'm sorry, I couldn't understand which driver you want to book.", "failed_node": "book_driver_node"}

        if not target_driver:
            logger.warning("Could not find a matching driver to book.")
            available_names = ", ".join([driver.name for driver in all_drivers[:3]])  # Show first 3 names
            return {
                "last_error": f"I couldn't find that specific driver to book. Available drivers include: {available_names}. Please be more specific.",
                "failed_node": "book_driver_node"
            }

        logger.info(f"Confirming booking with driver: {target_driver.name} (ID: {target_driver.id})")

        # 3. Call the tool to get booking confirmation details
        try:
            tool_response = await self.driver_tools.book_or_confirm_ride_with_driver.ainvoke({
                "city": state["search_city"],
                "page": state["current_page"],
                "driverId": target_driver.id,
            })

            if tool_response.get("success"):
                logger.info(f"Successfully retrieved booking details for driver {target_driver.id}.")
                return {
                    "booking_status": "confirmed",
                    "booking_details": tool_response,
                    "selected_driver": target_driver,  # Store the selected driver
                    "last_error": None,
                }
            else:
                error_msg = tool_response.get('error', 'An unknown error occurred.')
                logger.error(f"Booking tool failed: {error_msg}")
                return {"last_error": tool_response.get("msg", error_msg), "failed_node": "book_driver_node"}
        except Exception as e:
            logger.critical(f"A critical error occurred in BookDriverNode: {e}", exc_info=True)
            return {"last_error": "A system error occurred while confirming your booking.", "failed_node": "book_driver_node"}
