"""
Driver Details Node for the cab booking agent.
Handles showing detailed information about a specific driver.
"""

from typing import Dict, Any, List, Optional
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
import logging

from src.models.agent_state_model import AgentState
from src.models.drivers_model import DriverModel
from src.lngraph.tools.driver_tools import DriverTools

logger = logging.getLogger(__name__)


class DriverDetailsNode:
    """
    Node responsible for:
    1. Retrieving specific driver details
    2. Extracting requested information (images, vehicles, etc.)
    3. Formatting detailed driver information
    4. Handling specific attribute queries
    """

    def __init__(self, llm: BaseChatModel, driver_tools: DriverTools):
        """
        Initialize the driver details node.

        Args:
            llm: Language model for generating responses
            driver_tools: Driver tools instance for API calls
        """
        self.llm = llm
        self.driver_tools = driver_tools

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Show details for a specific driver.

        Args:
            state: Current agent state

        Returns:
            Updated state with driver details
        """
        try:
            # Get driver reference from state
            selected_driver = state.get("selected_driver")
            if not selected_driver:
                # This shouldn't happen if entry node works correctly
                logger.error("No driver selected for details")
                error_msg = await self._generate_no_driver_error(
                    state.get("conversation_language", "english")
                )
                state["messages"].append(AIMessage(content=error_msg))
                state["next_node"] = "wait_for_user_input"
                return state

            # Get the specific information requested
            requested_info = await self._extract_requested_info(
                state.get("last_user_message", ""),
                state.get("conversation_language", "english"),
            )

            # Get full driver details
            city = state.get("pickup_city")
            page = state.get("current_page", 1)

            try:
                result = await self.driver_tools.get_driver_info_tool(
                    city=city, page=page, driverId=selected_driver.id
                )

                if result["success"]:
                    driver = result["driver"]

                    # Generate appropriate response based on requested info
                    if requested_info["type"] == "specific":
                        response_message = await self._generate_specific_info_response(
                            driver,
                            requested_info["attributes"],
                            state.get("conversation_language", "english"),
                        )
                    else:
                        # General driver info request
                        response_message = await self._generate_full_details_response(
                            driver, state.get("conversation_language", "english")
                        )

                    # Update selected driver with fresh data
                    state["selected_driver"] = driver

                else:
                    # Use cached driver data if API fails
                    logger.warning("Failed to get fresh driver data, using cached")
                    driver = selected_driver

                    response_message = await self._generate_full_details_response(
                        driver, state.get("conversation_language", "english")
                    )

            except Exception as e:
                logger.error(f"Error getting driver details: {e}")
                # Use cached data
                driver = selected_driver
                response_message = await self._generate_full_details_response(
                    driver, state.get("conversation_language", "english")
                )

            # Add response to messages
            state["messages"].append(AIMessage(content=response_message))

            # Update driver history
            if "driver_history" not in state:
                state["driver_history"] = []
            if driver.id not in state["driver_history"]:
                state["driver_history"].append(driver.id)

            # Set next node
            state["next_node"] = "wait_for_user_input"

            return state

        except Exception as e:
            logger.error(f"Error in driver details node: {str(e)}")
            state["last_error"] = f"Failed to get driver details: {str(e)}"
            state["next_node"] = "error_handler_node"
            return state

    async def _extract_requested_info(
        self, user_message: str, language: str
    ) -> Dict[str, Any]:
        """
        Extract what specific information user wants about the driver.

        Args:
            user_message: User's message
            language: Conversation language

        Returns:
            Dictionary with requested info type and attributes
        """
        prompt = f"""Analyze what information the user wants about the driver.

                    User message: "{user_message}"

                    Identify if they want:
                        1. Specific attributes (car images, driver photo, vehicles, languages, etc.)
                        2. General information (full details)

                    Return JSON:
                        {{
                            "type": "specific" or "general",
                            "attributes": ["list", "of", "requested", "attributes"] or []
                        }}

                    Common specific requests:
                        - "car images", "vehicle photos" -> ["vehicle_images"]
                        - "driver photo", "profile picture" -> ["profile_image", "photos"]
                        - "what cars", "vehicles" -> ["vehicles"]
                        - "languages" -> ["languages"]
                        - "experience", "how long driving" -> ["experience", "connections"]
                        - "verified" -> ["verification_status"]
                        - "contact" -> ["phone", "profile_url"]

                    Return only the JSON.
                """

        try:
            response = await self.llm.ainvoke(prompt)
            import json

            result_text = response.content.strip()

            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            return json.loads(result_text)
        except Exception as e:
            logger.warning(f"Failed to extract requested info: {e}")
            return {"type": "general", "attributes": []}

    async def _generate_specific_info_response(
        self, driver: DriverModel, attributes: List[str], language: str
    ) -> str:
        """
        Generate response for specific driver attributes.

        Args:
            driver: Driver model
            attributes: Requested attributes
            language: Response language

        Returns:
            Response message
        """
        # Extract requested information
        info_dict = {}

        for attr in attributes:
            if attr in ["vehicle_images", "car_images"]:
                images = []
                for vehicle in driver.verified_vehicles:
                    for img in vehicle.images:
                        if img.full and img.full.url:
                            images.append(
                                {
                                    "vehicle": vehicle.model,
                                    "url": img.full.url,
                                    "type": img.type or "vehicle",
                                }
                            )
                info_dict["vehicle_images"] = images

            elif attr in ["profile_image", "photos", "driver_photo"]:
                info_dict["profile_image"] = driver.profile_image
                photos = []
                for photo in driver.photos:
                    if photo.full and photo.full.url:
                        photos.append(photo.full.url)
                info_dict["additional_photos"] = photos

            elif attr == "vehicles":
                vehicles = []
                for v in driver.verified_vehicles:
                    vehicles.append(
                        {
                            "model": v.model,
                            "type": v.vehicle_type,
                            "per_km_cost": v.per_km_cost,
                            "fuel_type": v.fuel_type,
                        }
                    )
                info_dict["vehicles"] = vehicles

            elif attr == "languages":
                info_dict["languages"] = driver.verified_languages

            elif attr in ["experience", "connections"]:
                info_dict["experience"] = f"{driver.experience} years"
                info_dict["total_rides"] = driver.connections

            elif attr == "verification_status":
                info_dict["profile_verified"] = driver.profile_verified
                info_dict["aadhar_verified"] = driver.aadhar_card_verified

            elif attr in ["phone", "contact", "profile_url"]:
                info_dict["contact"] = {
                    "name": driver.name,
                    "phone": driver.phone_no,
                    "profile_url": driver.constructed_profile_url,
                }

        prompt = f"""Generate a response showing specific driver information.

                    Driver: {driver.name}
                    Requested information: {attributes}
                    Extracted data: {info_dict}
                    Language: {language}

                    Requirements:
                        1. Present the requested information clearly
                        2. If showing images, mention them with descriptions
                        3. Format URLs and phone numbers properly
                        4. Use {language} language
                        5. Be concise but complete
                        6. If some info not available, mention it
                        7. Suggest next actions (book, ask more, etc.)

                    Generate only the response message.
                """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"Here's {driver.name}'s information: {info_dict}"

    async def _generate_full_details_response(
        self, driver: DriverModel, language: str
    ) -> str:
        """
        Generate comprehensive driver details response.

        Args:
            driver: Driver model
            language: Response language

        Returns:
            Response message
        """
        # Prepare comprehensive driver summary
        vehicles_summary = []
        for v in driver.verified_vehicles:
            vehicles_summary.append(
                f"{v.model} ({v.vehicle_type}) - ₹{v.per_km_cost}/km"
            )

        driver_summary = {
            "name": driver.name,
            "age": driver.age if driver.age > 0 else "Not specified",
            "gender": driver.gender,
            "experience": f"{driver.experience} years",
            "total_rides": driver.connections,
            "languages": ", ".join(driver.verified_languages),
            "vehicles": vehicles_summary,
            "verification": {
                "profile": "✓" if driver.profile_verified else "✗",
                "aadhar": "✓" if driver.aadhar_card_verified else "✗",
            },
            "services": {
                "pets_allowed": "Yes" if driver.is_pet_allowed else "No",
                "handicapped_friendly": "Yes"
                if driver.allow_handicapped_persons
                else "No",
                "personal_car_driving": "Yes"
                if driver.available_for_customers_personal_car
                else "No",
                "event_driving": "Yes"
                if driver.available_for_driving_in_event_wedding
                else "No",
            },
            "profile_url": driver.constructed_profile_url,
            "last_active": driver.last_access,
        }

        prompt = f"""Generate a comprehensive driver profile summary.

                Driver details: {driver_summary}
                Language: {language}

                Requirements:
                    1. Present information in an organized, easy-to-read format
                    2. Highlight important details (experience, verification, vehicles)
                    3. Group related information together
                    4. Mention special services if applicable
                    5. Use {language} language
                    6. Keep professional but friendly tone
                    7. End with call-to-action (view profile, book, ask questions)

                    Generate only the response message.
            """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            # Fallback
            if language == "hindi":
                return f"{driver.name} की जानकारी: {driver.experience} साल का अनुभव, {driver.connections} राइड्स पूरी की हैं। वाहन: {', '.join(vehicles_summary)}"
            else:
                return f"**{driver.name}**\nExperience: {driver.experience} years | Rides: {driver.connections}\nVehicles: {', '.join(vehicles_summary)}\nLanguages: {', '.join(driver.verified_languages)}"

    async def _generate_no_driver_error(self, language: str) -> str:
        """
        Generate error message when no driver is selected.

        Args:
            language: Response language

        Returns:
            Error message
        """
        prompt = f"""Generate a helpful message when user asks for driver details but no driver is selected.

                    Language: {language}

                    Requirements:
                        1. Politely explain no driver is selected
                        2. Suggest selecting a driver first
                        3. Offer to show available drivers
                        4. Use {language} language
                        5. Keep it brief and helpful

                        Generate only the message.
                """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except:
            if language == "hindi":
                return "कृपया पहले एक ड्राइवर चुनें। मैं आपको उपलब्ध ड्राइवर दिखा सकता हूं।"
            else:
                return "Please select a driver first. I can show you the available drivers if you'd like."
