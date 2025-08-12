# services/api_client.py
"""Simple API client for driver endpoints"""

import requests
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timezone
import json

import config

logger = logging.getLogger(__name__)


def get_drivers(
    city: str,
    page: int = 1,
    limit: int = config.DRIVERS_PER_FETCH,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Get drivers from the new single API endpoint"""
    try:
        params = {
            "city": city,
            "page": page,
            "limit": limit,
        }
        if filters:
            params.update(filters)

        response = requests.get(config.GET_PREMIUM_DRIVERS_URL, params=params, timeout=20)

        if response.status_code != 200:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return []

        result = response.json()
        if not result.get("success", False):
            logger.warning(f"API returned success=false for city {city}")
            return []

        return result.get("data", [])

    except Exception as e:
        logger.error(f"Error getting drivers from new API: {e}")
        return []


def create_trip(
    customer_details: Dict[str, str],
    pickup_city: str,
    drop_city: str,
    trip_type: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Create a trip using the new API endpoint with the updated payload."""
    try:
        payload = {
            "customerId": customer_details.get("id"),
            "customerName": customer_details.get("name"),
            "customerPhone": customer_details.get("phone"),
            "customerProfileImage": customer_details.get("profile_image"),
            "pickUpLocation": {
                "city": pickup_city,
                "coordinates": "",
                "placeName": "",
            },
            "dropLocation": {
                "city": drop_city,
                "coordinates": "",
                "placeName": "",
            },
            "startDate": start_date,
            "tripType": trip_type,
        }
        if end_date:
            payload["endDate"] = end_date

        response = requests.post(config.CREATE_TRIP_URL, json=payload, timeout=20)

        if response.status_code not in [200, 201]:
            logger.error(f"API error creating trip: {response.status_code} - {response.text}")
            return None

        return response.json()

    except Exception as e:
        logger.error(f"Error calling create_trip API: {e}")
        return None


def send_availability_request(
    trip_id: str,
    driver_ids: List[str],
    trip_details: Dict[str, Any],
    customer_details: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Sends an availability request to the specified drivers for a given trip."""
    try:
        payload = {
            "driverIds": ["pv258iLSjtfyBHyLgRjQcShJDt92", "NewcOnEO5DdiDkhKwc8LjGapICB3", "sHRv1ZZJ3pWKqH2yAo8OhRkwZPn2", "QKCrCXmknFc1ySeFrlwXUfCVYL93"],
            "data": {
                "trip_details": trip_details,
                "customerDetails": {
                    "name": customer_details.get("name"),
                    "id": customer_details.get("id"),
                    "phoneNo": customer_details.get("phone"),
                    "profile_image": customer_details.get("profile_image"),
                },
                "message": "Please confirm your availability for this trip.",
            },
            "tripId": trip_id,
        }

        # Detailed logging of the payload
        logger.info(f"Sending availability request. Trip ID: {trip_id}")
        logger.info(f"Payload for availability request: {json.dumps(payload, indent=2)}")

        response = requests.post(
            config.SEND_AVAILABILITY_REQUEST_URL, json=payload, timeout=20
        )

        if response.status_code not in [200, 201]:
            logger.error(
                f"API error sending availability request: {response.status_code} - {response.text}"
            )
            return None

        logger.info(f"Availability request sent successfully for Trip ID: {trip_id}")
        return response.json()

    except Exception as e:
        logger.error(f"Error sending availability request: {e}")
        return None
