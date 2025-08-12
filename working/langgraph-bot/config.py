# config.py
"""Simple configuration file"""

import os

# API Configuration
BASE_URL = "https://us-central1-cabswale-ai.cloudfunctions.net"
GET_PREMIUM_DRIVERS_URL = f"{BASE_URL}/cabbot-botApiGetPremiumDrivers"
CREATE_TRIP_URL = "https://cabbot-botcreatetrip-x7ozexvczq-uc.a.run.app"
SEND_AVAILABILITY_REQUEST_URL = "https://us-central1-cabswale-ai.cloudfunctions.net/cabbot-botSendAvilabilityRequestToDrivers"


# Environment
PORT = int(os.environ.get("PORT", 8000))
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")


# Driver fetching configuration
DRIVERS_PER_FETCH = 20  # Fetch 20 drivers at a time
DRIVERS_PER_DISPLAY = 5  # Show 5 drivers at a time
MAX_TOTAL_DRIVERS = 100  # Maximum 100 drivers per user (5 fetches)
MAX_FETCH_DEPTH = 5  # Maximum 5 fetches per search