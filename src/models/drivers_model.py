from pydantic import BaseModel, model_validator, Field, ConfigDict
from typing import Any, List, Optional, Literal, Dict


class Vehicle_Full_Image_Model(BaseModel):
    url: str
    type: str = "full"


class Vehicle_Image_Model(BaseModel):
    type: str
    verified: bool
    full: Vehicle_Full_Image_Model

    class Config:
        extra = "ignore"


class Vehicle_Model(BaseModel):
    is_commercial: Optional[bool] = Field(False, alias="is_commercial")
    images: List[Vehicle_Image_Model]
    fuel_type: Optional[str] = Field(None, alias="fuelType")
    model: str
    reg_no: str = Field(alias="reg_no")
    vehicle_type: str = Field(alias="vehicleType")
    per_km_cost: float = Field(alias="perKmCost")

    class config:
        populate_by_name = True
        extra = "ignore"


class Premium_driver_model(BaseModel):
    id: str
    name: Optional[str] = None
    city: Optional[str] = None
    phone_no: str = Field(alias="phoneNo")
    profile_image: Optional[str] = Field(None, alias="profile_image")
    username: str = Field(alias="userName")
    verified_vehicles: List[Vehicle_Model] = Field(alias="verifiedVehicles")
    preferences: Dict[str, bool]

    class config:
        populate_by_name = True
        extra = "ignore"

    @model_validator(mode="before")
    @classmethod
    def create_preferences_dict(cls, data: Any) -> Any:
        if isinstance(data, dict):
            preferences = {}
            preferences["isPetAllowed"] = data.pop("isPetAllowed", False)
            data["preferences"] = preferences
        return data


class Driver_Full_Photo_Model(BaseModel):
    type: str = "full"
    url: str


class Driver_Photos_Model(BaseModel):
    verified: bool
    full: Driver_Full_Photo_Model

    model_config = ConfigDict(extra="ignore")


class Driver_Model(BaseModel):
    existing_info: Premium_driver_model
    age: int
    bio: Optional[str] = Field(default="", alias="driverBio")
    connections: int
    connections_by_states: Dict[str, int] = Field(alias="connectionsStateCount")
    instagram_url: Optional[str] = Field(default="", alias="instagramUrl")
    languages: List[str]
    name: str
    notification_locations: Optional[List[str]] = Field(
        default=None, alias="notificationLocations"
    )
    married: bool
    photos: List[Driver_Photos_Model]
    profile_pic: str
    profile_url: Optional[str] = Field(default=None)
    trip_types: List[str] = Field(alias="tripTypes")
    username: str = Field(alias="userName")
    vehicle_ownership_details: List[bool] = Field(alias="vehicleOwnershipDetails")
    verified_languages: List[str] = []

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    class config:
        populate_by_name = True
        extra = "ignore"

    @model_validator(mode="before")
    @classmethod
    def extract_data(cls, data: Any) -> Any:
        """Extract data from API response structure"""
        if isinstance(data, dict):
            # If the data is wrapped in success/data structure
            if "success" in data and "data" in data:
                data = data["data"]
        return data

    @model_validator(mode="before")
    @classmethod
    def extract_profile_pic_url(cls, data: Any) -> Any:
        """Extract profile pic URL from nested structure"""
        if isinstance(data, dict):
            # Handle profile pic extraction
            if "profilePic" in data and isinstance(data["profilePic"], dict):
                # Extract the thumbnail URL from the nested structure
                if "thumb" in data["profilePic"] and isinstance(
                    data["profilePic"]["thumb"], dict
                ):
                    data["profile_pic"] = data["profilePic"]["thumb"].get("url", "")
                elif "mob" in data["profilePic"] and isinstance(
                    data["profilePic"]["mob"], dict
                ):
                    data["profile_pic"] = data["profilePic"]["mob"].get("url", "")
                else:
                    data["profile_pic"] = ""
            elif "profilePic" in data and isinstance(data["profilePic"], str):
                data["profile_pic"] = data["profilePic"]

            # Extract verified languages names
            if "verifiedLanguages" in data and isinstance(
                data["verifiedLanguages"], list
            ):
                verified_langs = []
                for lang in data["verifiedLanguages"]:
                    if isinstance(lang, dict) and "name" in lang:
                        verified_langs.append(lang["name"])
                    elif isinstance(lang, str):
                        verified_langs.append(lang)
                data["verified_languages"] = verified_langs

        return data

    @model_validator(mode="after")
    def set_profile_url(self) -> "Driver_Model":
        """Set profile URL if not provided"""
        if not self.profile_url and self.username:
            self.profile_url = f"www.cabswale.ai/profile/{self.username}"
        return self


__all__ = ["Driver_Model", "Premium_driver_model"]
