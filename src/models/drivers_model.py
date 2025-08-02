from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, computed_field, ConfigDict


class PhotoUrl(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    url: str


class Photo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mob: Optional[PhotoUrl] = Field(None)
    thumb: Optional[PhotoUrl] = Field(None)
    verified: Optional[bool] = Field(False)
    uploaded_at: Optional[Dict[str, int]] = Field(None, alias="uploadedAt")
    full: Optional[PhotoUrl] = Field(None)


class VehicleImage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mob: Optional[PhotoUrl] = Field(None)
    thumb: Optional[PhotoUrl] = Field(None)
    verified: Optional[bool] = Field(False)
    error_message: Optional[str] = Field(None, alias="errorMessage")
    uploaded_at: Optional[str] = Field(None, alias="uploadedAt")
    id: Optional[str] = Field(None)
    type: Optional[str] = Field(None)
    full: Optional[PhotoUrl] = Field(None)


class VerifiedVehicle(BaseModel):
    model_config = ConfigDict(extra="ignore")

    images: List[VehicleImage]
    reg_no: str
    fuel_type: Optional[str] = Field(None, alias="fuelType")
    per_km_cost: float = Field(alias="perKmCost")
    model: str
    video: Optional[Any] = None
    is_commercial: Optional[bool] = Field(None)
    vehicle_type: str = Field(alias="vehicleType")


class Membership(BaseModel):
    model_config = ConfigDict(extra="ignore")

    plan: Optional[str] = Field(None)
    duration: Optional[int] = Field(None)
    end_date: Optional[str] = Field(None, alias="endDate")


class DriverModel(BaseModel):
    id: str
    name: str
    phone_no: str = Field(alias="phoneNo")
    username: str = Field(alias="userName")
    city: str
    profile_image: str = Field(alias="profileImage")
    photos: List[Photo]
    verified_vehicles: List[VerifiedVehicle] = Field(alias="verifiedVehicles")
    notification_locations: List[str] = Field(alias="notificationLocations")
    from_top_routes: List[str] = Field(alias="fromTopRoutes")
    to_top_routes: List[str] = Field(alias="toTopRoutes")
    verified_languages: List[str] = Field(alias="verifiedLanguages")
    last_access: str = Field(alias="lastAccess")
    location_updated_at: str = Field(alias="locationUpdatedAt")
    is_pet_allowed: bool = Field(alias="isPetAllowed")
    aadhar_card_verified: bool = Field(alias="aadharCardVerified")
    verified: bool
    profile_verified: bool = Field(alias="profileVerified")
    is_completed: bool = Field(alias="isCompleted")
    premium_driver: bool = Field(alias="premiumDriver")
    allow_handicapped_persons: bool = Field(alias="allowHandicappedPersons")
    available_for_customers_personal_car: bool = Field(
        alias="availableForCustomersPersonalCar"
    )
    available_for_driving_in_event_wedding: bool = Field(
        alias="availableForDrivingInEventWedding"
    )
    available_for_part_time_full_time: bool = Field(
        alias="availableForPartTimeFullTime"
    )
    auto_approve_leads: bool = Field(alias="autoApproveLeads")
    condition_accepted: bool = Field(alias="conditionAccepted")
    married: bool
    notification_alert: bool = Field(alias="notificationAlert")
    nearby_notification_alert: bool = Field(alias="nearbyNotificationAlert")
    pause_nearby_notifications: bool = Field(alias="pauseNearbyNotifications")
    age: Optional[int] = Field(-1)
    connections: int
    experience: int
    driving_license_experience: int = Field(alias="drivingLicenseExperience")
    profile_completion_percentage: int = Field(alias="profileCompletionPercentage")
    profile_visits: int = Field(alias="profileVisits")
    incoming_calls: int = Field(alias="incomingCalls")
    outgoing_calls: int = Field(alias="outgoingCalls")
    messages_received: int = Field(alias="messagesReceived")
    messages_sent: int = Field(alias="messagesSent")
    total_leads: int = Field(alias="totalLeads")
    fraud_reports: int = Field(alias="fraudReports")
    gender: str
    identity: str
    driver_id: str = Field(alias="driverId")
    agent_id: Optional[str] = Field(None, alias="agentId")
    notification_place: str = Field(alias="notificationPlace")
    profile_url: str = Field(alias="profileUrl", default="")
    qr_code_url: str = Field(alias="qrCodeUrl")
    membership: Membership
    metadata_photos_arrangement: List[Any] = Field(alias="metadataPhotosArrangement")
    metadata_prominent_indexes: List[Any] = Field(alias="metadataProminentIndexes")
    current_location: List[float] = Field(alias="currentLocation")
    location: List[Any]

    @computed_field
    def constructed_profile_url(self) -> str:
        """Construct the profile URL from username"""
        return f"https://cabswale.ai/profile/{self.username}"

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class Pagination(BaseModel):
    page: int
    limit: int
    total: int
    has_more: bool = Field(alias="hasMore")


class SearchCoordinates(BaseModel):
    lat: float
    lng: float


class Search(BaseModel):
    city: str
    coordinates: SearchCoordinates
    radius: str
    strategy: str
    filters: Dict[str, Any]
    sort_by: str = Field(alias="sortBy")
    query_by: str = Field(alias="queryBy")


class APIResponse(BaseModel):
    success: bool
    data: List[DriverModel]
    pagination: Pagination
    search: Search
