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
        selected_driver = state.get("selected_driver")
        driver_summary = state.get("driver_summary")
        booking_details = state.get("booking_details")
        booking_status = state.get("booking_status")
        active_filters = state.get("active_filters", {})
        is_filtered = state.get("is_filtered", False)

        # Check if there's an error to address first
        if last_error:
            ai_message = AIMessage(content=last_error)
            return {"messages": [ai_message]}

        # Handle different response scenarios based on state
        response_content = ""

        # Booking confirmation response - HIGHEST PRIORITY
        if booking_status == "confirmed" and booking_details:
            driver_name = booking_details.get("Driver Name", "the driver")
            phone = booking_details.get("PhoneNo.", "Contact information not available")
            profile = booking_details.get("Profile", "")

            response_content = f"🎉 Great! Your booking with {driver_name} is confirmed!\n\n"
            response_content += f"📞 Contact Number: {phone}\n"
            if profile:
                response_content += f"👤 Profile: {profile}\n"
            response_content += "\n✅ The driver will contact you soon for pickup details. Have a safe trip!"

        # Driver info response with contact details if available
        elif selected_driver and driver_summary:
            name = driver_summary.get("name", "Unknown")
            age = driver_summary.get("age", "Not specified")
            city = driver_summary.get("city", "Unknown")
            experience = driver_summary.get("experience", 0)
            vehicles = driver_summary.get("vehicles", [])
            cost = driver_summary.get("avg_cost_per_km", 0)
            languages = driver_summary.get("languages", [])
            pet_allowed = driver_summary.get("pet_allowed", False)
            phone = driver_summary.get("phone", "")

            vehicle_text = ", ".join(vehicles) if vehicles else "unknown vehicle"
            language_text = ", ".join(languages) if languages else "not specified"
            pet_text = "allows pets" if pet_allowed else "doesn't allow pets"

            response_content = f"{name} is {age} years old from {city} with {experience} years of experience. "
            response_content += f"He drives {vehicle_text} and charges around ₹{cost} per km. "
            response_content += f"He speaks {language_text} and {pet_text}."

            if phone:
                response_content += f"\n\n📞 Contact: {phone}"

            if driver_summary.get("profile_url"):
                response_content += f"\n👤 Profile: {driver_summary['profile_url']}"

        # Driver search results response
        elif current_drivers:
            filter_text = ""
            if is_filtered and active_filters:
                filter_parts = []
                if active_filters.get("vehicle_types"):
                    filter_parts.append(f"vehicle: {', '.join(active_filters['vehicle_types'])}")
                if active_filters.get("min_experience"):
                    filter_parts.append(f"experience: {active_filters['min_experience']}+ years")
                if active_filters.get("married") is not None:
                    filter_parts.append(f"married: {'yes' if active_filters['married'] else 'no'}")
                if active_filters.get("gender"):
                    filter_parts.append(f"gender: {active_filters['gender']}")

                if filter_parts:
                    filter_text = f" (filtered by: {', '.join(filter_parts)})"

            response_content = f"I found {len(current_drivers)} driver{'s' if len(current_drivers) != 1 else ''} in {search_city}{filter_text}:\n\n"

            for i, driver in enumerate(current_drivers, 1):
                vehicle_types = []
                total_cost = 0
                vehicle_count = len(driver.verified_vehicles)

                for vehicle in driver.verified_vehicles:
                    vehicle_types.append(vehicle.vehicle_type)
                    total_cost += vehicle.per_km_cost

                avg_cost = total_cost / vehicle_count if vehicle_count > 0 else 0
                vehicle_text = ", ".join(set(vehicle_types)) if vehicle_types else "unknown"

                # Use actual experience from the driver model
                experience = driver.experience if hasattr(driver, 'experience') else 0

                response_content += f"{i}. {driver.name} ({experience} yrs exp, {vehicle_text}, ₹{avg_cost:.0f}/km)\n"

            response_content += "\nYou can ask me about any specific driver or book one of them!"

        # Default response for missing information
        else:
            if not state.get("pickupLocation") or not state.get("dropLocation"):
                response_content = "To help you book a cab, I need to know:\n"
                if not state.get("pickupLocation"):
                    response_content += "• Where should we pick you up?\n"
                if not state.get("dropLocation"):
                    response_content += "• Where are you going?\n"
            elif not search_city:
                response_content = "Which city are you looking for a cab in?"
            else:
                response_content = "How can I help you with booking a cab today?"

        try:
            # Use the response content directly without LLM processing to avoid modifications
            ai_message = AIMessage(content=response_content)
            logger.info(f"Generated AI Response: {response_content}")
            return {"messages": [ai_message]}

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            error_message = AIMessage(content="I'm sorry, I encountered an issue while generating a response. Could you please try again?")
            return {"messages": [error_message]}
