from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class BookDriverInput(BaseModel):
    """Input Schema for Driver booked or chosen by user"""

    city: str = Field(
        ...,
        description="City where the driver operates or currently in (required for cache key generation)",
    )

    page: int = Field(
        ...,
        ge=1,
        description="Page number where this driver was found (required for cache retrieval)",
    )

    driverId: str = Field(
        ..., description="Unique identifier of the driver (e.g., 'DRV123456')"
    )


class FilterDriverInput(BaseModel):
    """Input schema for filtering drivers based on user requirements"""

    city: str = Field(
        ...,
        description="City name where drivers are being searched (e.g., 'delhi', 'mumbai')",
    )

    page: int = Field(
        default=1, ge=1, description="Page number for pagination. Defaults to 1"
    )

    filter_obj: Dict[str, Any] = Field(
        ...,
        description="""
        Dictionary containing filter criteria. Supported filters:
        - vehicle_types: List[str] - Types of vehicles (e.g., ["sedan", "hatchback", "suv"])
        - gender: str - Driver gender ("male" or "female")
        - min_age: int - Minimum driver age
        - max_age: int - Maximum driver age
        - is_pet_allowed: bool - Whether driver allows pets
        - min_connections: int - Minimum number of connections/rides completed
        - min_experience: int - Minimum years of driving experience
        - languages: List[str] - Languages spoken by driver
        - profile_verified: bool - Whether driver's profile is verified
        - married: bool - Driver's marital status
        - allow_handicapped_persons: bool - Accommodates handicapped passengers
        - available_for_customers_personal_car: bool - Available to drive customer's car
        - available_for_driving_in_event_wedding: bool - Available for event/wedding driving
        - available_for_part_time_full_time: bool - Available for part/full-time work

        Example: {"vehicle_types": ["sedan", "suv"], "gender": "female", "min_experience": 3}
        """,
    )


class DriverInfoInput(BaseModel):
    """Input schema for retrieving specific driver information"""

    city: str = Field(
        ...,
        description="City where the driver operates or currently in (required for cache key generation)",
    )

    page: int = Field(
        ...,
        ge=1,
        description="Page number where this driver was found (required for cache retrieval)",
    )

    driverId: str = Field(
        ..., description="Unique identifier of the driver (e.g., 'DRV123456')"
    )


class SearchDriversInput(BaseModel):
    """Input schema for searching drivers with comprehensive filtering options"""

    city: str = Field(
        ...,
        description="City name for driver search (e.g., 'delhi', 'mumbai', 'bangalore')",
    )

    page: int = Field(
        default=1, ge=1, description="Page number for paginated results. Starts from 1"
    )

    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of drivers per page (min: 1, max: 100)",
    )

    radius: int = Field(
        default=100, ge=1, description="Search radius in kilometers from city center"
    )

    search_strategy: str = Field(
        default="hybrid",
        description="Search strategy: 'city' (city-based), 'geo' (location-based), or 'hybrid' (combined)",
    )

    sort_by: str = Field(
        default="lastAccess:desc",
        description="Sort criteria (e.g., 'lastAccess:desc', 'rating:desc', 'experience:desc')",
    )

    vehicle_types: Optional[List[str]] = Field(
        None,
        description="Filter by vehicle types (e.g., ['sedan', 'suv', 'hatchback', 'luxury'])",
    )

    gender: Optional[str] = Field(
        None, description="Filter by driver gender: 'male' or 'female'"
    )

    min_age: Optional[int] = Field(
        None, ge=18, description="Minimum driver age (must be 18+)"
    )

    max_age: Optional[int] = Field(None, le=70, description="Maximum driver age")

    is_pet_allowed: Optional[bool] = Field(
        None, description="Filter for pet-friendly drivers"
    )

    min_connections: Optional[int] = Field(
        None, ge=0, description="Minimum number of completed rides/connections"
    )

    min_experience: Optional[int] = Field(
        None, ge=0, description="Minimum years of driving experience"
    )

    languages: Optional[List[str]] = Field(
        None,
        description="Filter by languages spoken (e.g., ['hindi', 'english', 'punjabi'])",
    )

    profile_verified: Optional[bool] = Field(
        None, description="Filter for verified driver profiles only"
    )

    married: Optional[bool] = Field(None, description="Filter by marital status")

    custom_filters: Optional[Dict[str, Any]] = Field(
        None, description="Additional custom filters as key-value pairs"
    )

    use_cache: bool = Field(
        default=True, description="Whether to use cached results for faster response"
    )


class CreateTripInput(BaseModel):
    customer_id: Optional[str] = Field(None, description="ID of the customer")
    customer_name: Optional[str] = Field(None, description="Name of the customer")
    customer_phone: Optional[str] = Field(None, description="Phone number of the customer")
    customer_profile_image_url: Optional[str] = Field(None, description="URL of the customer's profile image")
    pickup_location: Optional[str] = Field(None, description="Pickup location")
    dropoff_location: str = Field(..., description="Dropoff location")
    trip_type: str = Field(..., description="Type of trip: 'one-way' or 'round-trip', or 'multi city' etc...")
    start_date: datetime = Field(..., description="Start date of the trip, in iso format. ex: 2023-09-25T12:00:00")
    end_date: datetime = Field(..., description="End date of the trip, in iso format. ex: 2023-09-25T12:00:00")
