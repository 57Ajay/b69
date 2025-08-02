"""
Simple working version of the cab booking agent.
This is a minimal implementation that works correctly.
"""

from src.lngraph.tools.driver_tools import DriverTools
from src.services.api_service import DriversAPIClient
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_vertexai import ChatVertexAI
import asyncio
import os
import sys
from typing import Dict, Any, List
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class SimpleCabAgent:
    """Simplified cab booking agent that works."""

    def __init__(self):
        """Initialize the agent."""
        print("Initializing agent components...")

        # Initialize LLM
        self.llm = ChatVertexAI(
            model="gemini-1.5-flash", temperature=0.7, max_output_tokens=1024
        )

        # Initialize API client
        self.api_client = DriversAPIClient("simple_session", 5)
        self.driver_tools = DriverTools(self.api_client)

        # Simple state
        self.state = {
            "messages": [],
            "pickup_city": None,
            "destination_city": None,
            "current_drivers": [],
            "selected_driver": None,
            "active_filters": {},
        }

        print("Agent ready!\n")

    async def process_message(self, message: str) -> str:
        """Process a user message and return response."""
        self.state["messages"].append(HumanMessage(content=message))

        # Extract intent and entities using LLM
        intent_prompt = f"""Analyze this user message and extract:
1. Intent: One of [search_drivers, filter_drivers, driver_info, book_driver, general]
2. City: Extract city name if mentioned (must be Indian city)
3. Filters: Extract any filters like vehicle type, gender, experience, etc.

User message: "{message}"

Respond in this exact format:
Intent: <intent>
City: <city or None>
Filters: <comma-separated filters or None>

Be precise and extract only what's explicitly mentioned."""

        try:
            response = await self.llm.ainvoke(intent_prompt)
            analysis = response.content

            # Parse response
            intent = "general"
            city = None
            filters = {}

            for line in analysis.split("\n"):
                if line.startswith("Intent:"):
                    intent = line.split(":", 1)[1].strip()
                elif line.startswith("City:"):
                    city_text = line.split(":", 1)[1].strip()
                    if city_text and city_text.lower() != "none":
                        city = city_text.lower()
                elif line.startswith("Filters:"):
                    filter_text = line.split(":", 1)[1].strip()
                    if filter_text and filter_text.lower() != "none":
                        # Parse filters
                        if "suv" in filter_text.lower():
                            filters["vehicle_types"] = ["suv"]
                        if "sedan" in filter_text.lower():
                            filters["vehicle_types"] = ["sedan"]
                        if "female" in filter_text.lower():
                            filters["gender"] = "female"
                        if (
                            "male" in filter_text.lower()
                            and "female" not in filter_text.lower()
                        ):
                            filters["gender"] = "male"

            # Handle based on intent
            if intent == "search_drivers":
                if city:
                    self.state["pickup_city"] = city
                    return await self.search_drivers(city, filters)
                elif self.state["pickup_city"]:
                    return await self.search_drivers(self.state["pickup_city"], filters)
                else:
                    return "Which city would you like to find drivers in?"

            elif intent == "filter_drivers":
                if self.state["current_drivers"] and filters:
                    return await self.apply_filters(filters)
                elif not self.state["current_drivers"]:
                    return "Let me first search for drivers. Which city are you in?"
                else:
                    return "What filters would you like to apply? You can filter by vehicle type (SUV, sedan), gender, experience, etc."

            elif intent == "driver_info":
                return await self.show_driver_info(message)

            elif intent == "book_driver":
                return await self.book_driver(message)

            else:
                # General response
                if city:
                    self.state["pickup_city"] = city
                    return await self.search_drivers(city, {})
                else:
                    return "I can help you find and book drivers in any Indian city. Just tell me which city you need a driver in!"

        except Exception as e:
            print(f"Error: {e}")
            return "I'm having trouble processing your request. Could you please try again?"

    async def search_drivers(self, city: str, filters: Dict[str, Any]) -> str:
        """Search for drivers in a city."""
        try:
            print(f"Searching drivers in {city}...")

            # Merge filters
            all_filters = {**self.state["active_filters"], **filters}
            self.state["active_filters"] = all_filters

            # Call API
            params = {"city": city, "page": 1, "limit": 10}

            # Add filters
            if "vehicle_types" in all_filters:
                params["vehicle_types"] = all_filters["vehicle_types"]
            if "gender" in all_filters:
                params["gender"] = all_filters["gender"]

            result = await self.driver_tools.search_drivers_tool(**params)

            if result["success"] and result["drivers"]:
                self.state["current_drivers"] = result["drivers"]

                # Format response
                response = f"I found {result['total']} drivers in {city.title()}"
                if all_filters:
                    filter_desc = []
                    if "vehicle_types" in all_filters:
                        filter_desc.append(
                            f"{', '.join(all_filters['vehicle_types']).upper()}"
                        )
                    if "gender" in all_filters:
                        filter_desc.append(f"{all_filters['gender']}")
                    response += f" (Filters: {', '.join(filter_desc)})"
                response += ". Here are the top matches:\n\n"

                # Show top 5 drivers
                for i, driver in enumerate(result["drivers"][:5]):
                    vehicles = ", ".join(
                        [v.vehicle_type for v in driver.verified_vehicles]
                    )
                    response += (
                        f"{i + 1}. **{driver.name}** - {driver.experience} years exp"
                    )
                    if vehicles:
                        response += f" | Vehicles: {vehicles}"
                    if driver.is_pet_allowed:
                        response += " | 🐕 Pet-friendly"
                    response += f"\n"

                if result["total"] > 5:
                    response += (
                        f"\n... and {result['total'] - 5} more drivers available."
                    )

                response += "\n\nYou can:\n• Ask for more details about any driver\n• Apply filters (vehicle type, gender, etc.)\n• Book a driver"

                return response
            else:
                self.state["current_drivers"] = []
                return f"No drivers found in {city.title()}. Try searching in a nearby city or remove filters."

        except Exception as e:
            print(f"Search error: {e}")
            return "I couldn't search for drivers right now. Please try again."

    async def apply_filters(self, filters: Dict[str, Any]) -> str:
        """Apply filters to current results."""
        if not self.state["pickup_city"]:
            return "Please tell me which city you'd like to search in first."

        # Update filters and search again
        self.state["active_filters"].update(filters)
        return await self.search_drivers(self.state["pickup_city"], {})

    async def show_driver_info(self, message: str) -> str:
        """Show information about a specific driver."""
        if not self.state["current_drivers"]:
            return "Please search for drivers first. Which city are you in?"

        # Find driver reference
        driver = None
        message_lower = message.lower()

        # Check for ordinal references
        ordinals = [
            "first",
            "1st",
            "second",
            "2nd",
            "third",
            "3rd",
            "fourth",
            "4th",
            "fifth",
            "5th",
        ]
        for i, ordinal in enumerate(ordinals):
            if ordinal in message_lower and i < len(self.state["current_drivers"]):
                driver = self.state["current_drivers"][i]
                break

        # Check for name
        if not driver:
            for d in self.state["current_drivers"]:
                if d.name.lower() in message_lower:
                    driver = d
                    break

        if driver:
            self.state["selected_driver"] = driver

            # Format detailed info
            vehicles_info = "\n".join(
                [
                    f"  - {v.model} ({v.vehicle_type}) - ₹{v.per_km_cost}/km"
                    for v in driver.verified_vehicles
                ]
            )

            response = f"""**Driver Details: {driver.name}**

📋 **Basic Info:**
• Age: {driver.age if driver.age > 0 else "Not specified"}
• Gender: {driver.gender.title()}
• Experience: {driver.experience} years
• Total Rides: {driver.connections}

🚗 **Vehicles:**
{vehicles_info}

✅ **Services:**
• Languages: {", ".join(driver.verified_languages)}
• Pet Friendly: {"Yes 🐕" if driver.is_pet_allowed else "No"}
• Profile Verified: {"Yes ✓" if driver.profile_verified else "No"}

📞 To book this driver, just say "Book {driver.name}" or "Book this driver"."""

            return response
        else:
            return "I couldn't find that driver. Please specify which driver you're interested in (e.g., 'first driver', 'second driver', or use their name)."

    async def book_driver(self, message: str) -> str:
        """Book a driver."""
        # Find driver to book
        driver = None
        message_lower = message.lower()

        # Check if user wants to book selected driver
        if self.state["selected_driver"] and any(
            word in message_lower for word in ["this", "him", "her"]
        ):
            driver = self.state["selected_driver"]
        else:
            # Try to find driver by name or position
            if self.state["current_drivers"]:
                # Check ordinals
                ordinals = ["first", "1st", "second", "2nd", "third", "3rd"]
                for i, ordinal in enumerate(ordinals):
                    if ordinal in message_lower and i < len(
                        self.state["current_drivers"]
                    ):
                        driver = self.state["current_drivers"][i]
                        break

                # Check by name
                if not driver:
                    for d in self.state["current_drivers"]:
                        if d.name.lower() in message_lower:
                            driver = d
                            break

        if driver:
            vehicle = driver.verified_vehicles[0] if driver.verified_vehicles else None
            vehicle_info = (
                f"{vehicle.model} ({vehicle.vehicle_type})"
                if vehicle
                else "Vehicle not specified"
            )

            response = f"""✅ **Booking Confirmed!**

**Driver Details:**
• Name: {driver.name}
• Phone: 📞 {driver.phone_no}
• Vehicle: {vehicle_info}

**Next Steps:**
1. Call the driver at {driver.phone_no}
2. Confirm your pickup location
3. Discuss the fare and trip details

💡 **Tips:**
• Save the driver's number for future reference
• Confirm the fare before starting the trip
• Share your trip details with family/friends

Thank you for using our service! Have a safe journey! 🚗"""

            return response
        else:
            if not self.state["current_drivers"]:
                return "Please search for drivers first. Which city do you need a driver in?"
            else:
                return "Please specify which driver you'd like to book (e.g., 'Book the first driver' or 'Book Raj Kumar')."

    async def cleanup(self):
        """Cleanup resources."""
        await self.api_client.close()


async def run_simple_cli():
    """Run the simple CLI interface."""
    print("\n" + "=" * 60)
    print("🚗 Simple Cab Booking Assistant")
    print("=" * 60)
    print("\nI can help you find and book drivers in any Indian city.")
    print("Just tell me where you need a ride!\n")

    agent = SimpleCabAgent()

    print("Examples:")
    print("- 'Find drivers in Delhi'")
    print("- 'Show me SUV drivers'")
    print("- 'Tell me about the first driver'")
    print("- 'Book Raj Kumar'\n")

    try:
        while True:
            # Get user input
            try:
                user_input = input("\nYou: ").strip()
            except KeyboardInterrupt:
                break

            if user_input.lower() in ["quit", "exit", "bye"]:
                break

            if not user_input:
                continue

            # Process message
            print("\nAssistant: ", end="", flush=True)
            response = await agent.process_message(user_input)
            print(response)

    finally:
        print("\n\n👋 Thank you for using Cab Booking Assistant!")
        await agent.cleanup()


if __name__ == "__main__":
    # Run the simple version
    try:
        asyncio.run(run_simple_cli())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
