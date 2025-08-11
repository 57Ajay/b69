from datetime import datetime
from zoneinfo import ZoneInfo

def get_ist_in_utc_iso():
    # Get current IST time
    ist_time = datetime.now(ZoneInfo("Asia/Kolkata"))

    # Convert to UTC
    utc_time = ist_time.astimezone(ZoneInfo("UTC"))

    # Format as 2025-08-06T09:00:00.000Z
    return utc_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

indian_time = get_ist_in_utc_iso()

__all__ = ["indian_time"]
