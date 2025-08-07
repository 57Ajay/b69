from re import A
from typing import Dict, Any
from src.models.agent_state_model import AgentState
import logging
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import AIMessage
from src.services.api_service import DriversAPIClient
logger = logging.getLogger(__name__)


class ResponseGeneratorNode:
    """
    FIXED: Node to generate a final, user-facing response based on the agent's state.
    Handles filtering, trip collection, and driver display properly.
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

    def _format_filter_summary(self, active_filters: Dict[str, Any]) -> str:
        """
        FIXED: Create human-readable filter summary.
        """
        if not active_filters:
            return ""

        filter_parts = []
        for key, value in active_filters.items():
            if key == "vehicle_types" and value:
                if isinstance(value, list):
                    filter_parts.append(f"vehicle types: {', '.join(value)}")
                else:
                    filter_parts.append(f"vehicle type: {value}")
            elif key == "married":
                filter_parts.append("married" if value else "unmarried")
            elif key == "min_age":
                filter_parts.append(f"min age: {value}")
            elif key == "min_experience":
                filter_parts.append(f"min experience: {value} years")
            elif key == "gender":
                filter_parts.append(f"gender: {value}")
            elif key == "languages" and value:
                if isinstance(value, list):
                    filter_parts.append(f"languages: {', '.join(value)}")
                else:
                    filter_parts.append(f"language: {value}")
            elif key == "is_pet_allowed":
                filter_parts.append("pet-friendly" if value else "no pets")
            else:
                filter_parts.append(f"{key.replace('_', ' ')}: {value}")

        return f" (filtered by: {', '.join(filter_parts)})" if filter_parts else ""

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        FIXED: Generates a response to the user based on the current state with proper filter handling.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary containing the new AI message to be added to the history.
        """
        logger.info("Executing ResponseGeneratorNode...")

        returnObj = {}

        # Get state variables
        current_drivers = state.get("current_drivers", [])
        search_city = state.get("search_city", "")
        page = state.get("current_page", 1)
        last_error = state.get("last_error")
        selected_driver = state.get("selected_driver")
        driver_summary = state.get("driver_summary")
        booking_details = state.get("booking_details")
        booking_status = state.get("booking_status")
        active_filters = state.get("active_filters", {})
        is_filtered = state.get("is_filtered", False)
        has_more_results = state.get("has_more_results", False)
        last_user_message = state.get("last_user_message", "").lower()

        # Handle errors first
        if last_error:
            ai_message = AIMessage(content=last_error)
            return {"messages": [ai_message]}

        # Handle booking confirmation - HIGHEST PRIORITY
        if booking_status == "confirmed" and booking_details:
            driver_name = booking_details.get("Driver Name", "the driver")
            phone = booking_details.get("PhoneNo.", "Contact information not available")
            profile = booking_details.get("Profile", "")

            response_content = f"🎉 Great! Your booking with {driver_name} is confirmed!\n\n"
            response_content += f"📞 Contact Number: {phone}\n"
            response_content += f"👤 Profile: {profile}\n"
            response_content += "\n✅ The driver will contact you soon for pickup details. Have a safe trip!"

            returnObj["booking_status"] = "none"

        # Handle specific driver info response
        elif selected_driver and driver_summary:
            name = driver_summary.get("name", "Unknown")
            vehicles = driver_summary.get("vehicles", [])

            # Check for specific information requests
            if "image" in last_user_message or "photo" in last_user_message:
                images = []
                for v in vehicles:
                    if "images: " in v:
                        img_part = v.split("images: ")[1]
                        if img_part.startswith("[") and img_part.endswith("]"):
                            img_part = img_part[1:-1]  # Remove brackets
                        images.extend([img.strip().strip("'\"") for img in img_part.split(",") if img.strip()])

                if images:
                    response_content = f"Here are the images for {name}'s vehicle:\n" + "\n".join(images[:3])  # Show first 3 images
                else:
                    response_content = f"I couldn't find any images for {name}'s vehicle."

            elif any(v_type in last_user_message for v_type in ["vehicle", "sedan", "suv", "hatchback", "car"]):
                vehicle_types = []
                for v in vehicles:
                    if "vehicle_type: " in v:
                        vehicle_types.append(v.split("vehicle_type: ")[1].split(" ")[0])

                if vehicle_types:
                    response_content = f"{name} drives: {', '.join(set(vehicle_types))}."
                else:
                    response_content = f"I don't have vehicle information for {name}."

            elif "married" in last_user_message or "marital" in last_user_message:
                response_content = f"{name} is {'married' if driver_summary.get('married') else 'not married'}."

            elif "profile" in last_user_message or "link" in last_user_message:
                profile_url = driver_summary.get('profile_url', '')
                if profile_url:
                    response_content = f"Here is {name}'s profile: {profile_url}"
                else:
                    response_content = f"I don't have a profile link for {name}."

            elif "experience" in last_user_message:
                exp = driver_summary.get('experience', 0)
                response_content = f"{name} has {exp} years of driving experience."

            elif "contact" in last_user_message or "phone" in last_user_message:
                phone = driver_summary.get('phone', '')
                if phone:
                    response_content = f"You can contact {name} at: {phone}"
                else:
                    response_content = f"I don't have contact information for {name}."

            else:
                age = driver_summary.get("age", "Not specified")
                city = driver_summary.get("city", "Unknown")
                experience = driver_summary.get("experience", 0)
                languages = driver_summary.get("languages", [])
                pet_allowed = driver_summary.get("pet_allowed", False)
                phone = driver_summary.get("phone", "")
                gender = driver_summary.get("gender", "")
                per_km_cost = driver_summary.get("per_km_cost", [0])

                vehicle_text = ", ".join(vehicles) if vehicles else "unknown vehicle"
                language_text = ", ".join(languages) if languages else "not specified"
                pet_text = "allows pets" if pet_allowed else "doesn't allow pets"
                pronoun = "She" if gender.lower() == "female" else "He"
                avg_cost = sum(per_km_cost) / len(per_km_cost) if per_km_cost else 0

                response_content = f"{name} is {age} years old from {city} with {experience} years of experience. "
                response_content += f"{pronoun} drives {vehicle_text} and charges around ₹{avg_cost:.0f} per km. "
                response_content += f"{pronoun} speaks {language_text} and {pet_text}."

                if phone:
                    response_content += f"\n\n📞 Contact: {phone}"
                if driver_summary.get("profile_url"):
                    response_content += f"\n👤 Profile: {driver_summary['profile_url']}"

            returnObj["driver_summary"] = None
            returnObj["current_driver"] = None

        # FIXED: Handle driver search results with proper filtering
        elif current_drivers:
            filter_text = self._format_filter_summary(active_filters) if is_filtered else ""

            response_content = f"I found {len(current_drivers)} driver{'s' if len(current_drivers) != 1 else ''} in {search_city}{filter_text}:\n\n"

            for i, driver in enumerate(current_drivers, 1):
                try:
                    if not search_city:
                        response_content += f"{i}. {driver['driver_name']} - Details unavailable\n"
                        continue

                    cache_key = self.client._generate_cache_key(search_city, page)
                    logger.debug(f"cache_key: {cache_key}, and search_city: {search_city}, and page: {page}")
                    full_driver_detail = await self.client._get_driver_detail(cache_key, driver["driver_id"])

                    logger.debug(f"full_driver_detail: {full_driver_detail}")

                    vehicle_types = [v.vehicle_type for v in full_driver_detail.verified_vehicles]
                    per_km_cost = [v.per_km_cost for v in full_driver_detail.verified_vehicles]
                    vehicle_text = ", ".join(set(vehicle_types)) or "unknown"
                    experience = full_driver_detail.experience if full_driver_detail.experience > 0 else full_driver_detail.experience + 1


                    response_content += f"{i}. {full_driver_detail.name} {experience} yrs exp, {vehicle_text}, ₹{per_km_cost}/km\n"

                except Exception as e:
                    logger.warning(f"Could not get details for driver {driver['driver_id']}: {e}")
                    response_content += f"{i}. {driver['driver_name']} - Details unavailable\n"

            # FIXED: Better call-to-action based on state
            if has_more_results:
                response_content += "\nSay 'show more' to see more drivers."
            elif not is_filtered:
                response_content += "\nYou can ask me about any specific driver, apply filters, or book one of them!"
            else:
                response_content += "\nYou can ask me about any specific driver, modify filters, or book one of them!"

            returnObj["current_driver"] = None
            returnObj["is_filtered"] = False

        # Handle no drivers found after filtering
        elif is_filtered and not current_drivers and active_filters:
            filter_summary = self._format_filter_summary(active_filters)
            response_content = f"No drivers found in {search_city}{filter_summary}. Would you like to remove some filters or try different criteria?"

        # FIXED: Better initial guidance
        else:
            pickup = state.get("pickupLocation")
            drop = state.get("dropLocation")
            trip_type = state.get("trip_type")

            if not pickup and not drop and not trip_type:
                response_content = "Hello! I'm here to help you book a cab. To get started, I need to know:\n\n"
                response_content += "• **Pickup location** (which city?)\n"
                response_content += "• **Destination** (where are you going?)\n"
                response_content += "• **Trip type** (one-way, round-trip, or multi-city)\n\n"
                response_content += "For example, you can say: 'I need a cab from Delhi to Jaipur for a one-way trip'"

            elif pickup and not drop:
                response_content = f"I see you want to travel from {pickup}. Where would you like to go? Please tell me your destination."

            elif not pickup and drop:
                response_content = f"I see you want to go to {drop}. Where will you be starting your journey from?"

            elif not pickup and not drop and not trip_type:
                response_content = f"ok, you want to travel from {pickup} to {drop}, can you provide me with the trip type? For example, you can say: 'I need a one-way trip or a round-trip'"

            elif not search_city:
                response_content = "I have your trip details but I'm not sure which city to search for drivers in. Could you clarify?"

            else:
                response_content = "How can I help you with your cab booking today? You can ask me to find drivers, apply filters, or get information about specific drivers."

        try:
            ai_message = AIMessage(content=response_content)
            logger.info(f"Generated AI Response: {response_content}")
            returnObj["messages"] = [ai_message]
            return returnObj

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            error_message = AIMessage(content="I'm sorry, I encountered an issue while generating a response. Could you please try again?")
            return {"messages": [error_message]}
