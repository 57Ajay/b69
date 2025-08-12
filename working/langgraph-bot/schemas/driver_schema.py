# schemas/driver_schema.py
"""Driver data schemas"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from langchain_core.messages import BaseMessage


class VehicleImageUrl(BaseModel):
    url: str


class VehicleImages(BaseModel):
    full: VehicleImageUrl


class Vehicles(BaseModel):
    reg_no: str
    model: str
    is_commercial: Optional[bool] = None
    per_km_cost: Optional[float] = None
    vehicle_type: str
    fuel_type: str
    images: List[VehicleImages]


class PremiumDriver(BaseModel):
    id: str
    name: Optional[str] = None
    city: Optional[str] = None
    phoneNo: str
    profile_image: Optional[str] = None
    username: Optional[str] = None
    verifiedVehicles: List[Vehicles]


class Routes(BaseModel):
    from_: str = Field(alias="from")
    to_: str


class Languages(BaseModel):
    name: str
    verified: bool


class TrainingContent(BaseModel):
    title: str
    description: str


class Driver(BaseModel):
    existingInfo: PremiumDriver
    age: Optional[int] = None
    connections: int = 0
    bio: Optional[str] = None
    experience: int = 0
    is_pet_allowed: Optional[bool] = None
    languages: List[str]
    is_married: Optional[bool] = None
    phoneNo: str
    routes: List[Routes]
    trip_types: List[str]
    username: Optional[str] = None
    trainingContent: List[TrainingContent] = []
    vehicle_ownership: List[bool]
    verified_languages: List[Languages]
    onboarded_at: Optional[datetime] = None


class CabBookingState(BaseModel):
    """Main state for the cab booking agent"""

    # Chat history
    chat_history: List[BaseMessage] = Field(default_factory=list)

    # Driver data - now storing all fetched drivers
    all_fetched_drivers: List[Dict[str, Any]] = Field(default_factory=list)
    drivers_with_full_details: List[Dict[str, Any]] = Field(default_factory=list)
    filtered_drivers: List[Dict[str, Any]] = Field(default_factory=list)

    # Pagination state
    current_display_index: int = 0
    current_page: int = 1
    fetch_count: int = 0

    # Filters
    applied_filters: Dict[str, Any] = Field(default_factory=dict)

    # Booking details
    pickup_location: Optional[str] = None
    drop_location: Optional[str] = None
    passenger_count: Optional[int] = None
    trip_type: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    trip_id: Optional[str] = None

    # State info
    last_bot_response: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)

    #customer details
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    customer_profile: Optional[str] = None
    customer_phone: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
