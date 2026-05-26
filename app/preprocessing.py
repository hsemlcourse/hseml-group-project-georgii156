from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pandas as pd

FEATURES = [
    "vendor_id",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "distance_km",
    "store_and_fwd_flag",
]


def haversine_distance_km(
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
) -> float:
    """Calculate distance between two points using the haversine formula."""
    earth_radius_km = 6371.0

    lat1 = math.radians(pickup_latitude)
    lon1 = math.radians(pickup_longitude)
    lat2 = math.radians(dropoff_latitude)
    lon2 = math.radians(dropoff_longitude)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius_km * c


def encode_vendor_id(vendor_id: int) -> int:
    """
    Reproduce category encoding from the notebook.

    In the notebook vendor_id was converted with:
    df["vendor_id"].astype("category").cat.codes

    For the NYC Taxi dataset usual vendor_id values are 1 and 2,
    therefore they become 0 and 1.
    """
    if vendor_id == 1:
        return 0
    if vendor_id == 2:
        return 1
    return int(vendor_id)


def encode_passenger_count(passenger_count: int) -> int:
    """
    Reproduce passenger_count encoding close to the notebook.

    In the original dataset passenger_count is numeric. The notebook used
    category codes, but for typical values 0..6 the code is effectively the
    same as the value. We keep it numeric for stable inference.
    """
    return int(passenger_count)


def encode_store_and_fwd_flag(value: str) -> int:
    return 1 if str(value).upper() == "Y" else 0


def prepare_features(data: dict[str, Any]) -> pd.DataFrame:
    """Convert one input request into a dataframe with the exact model columns."""
    pickup_datetime = pd.to_datetime(data["pickup_datetime"])

    pickup_latitude = float(data["pickup_latitude"])
    pickup_longitude = float(data["pickup_longitude"])
    dropoff_latitude = float(data["dropoff_latitude"])
    dropoff_longitude = float(data["dropoff_longitude"])

    row = {
        "vendor_id": encode_vendor_id(int(data["vendor_id"])),
        "passenger_count": encode_passenger_count(int(data["passenger_count"])),
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "hour": int(pickup_datetime.hour),
        "day_of_week": int(pickup_datetime.dayofweek),
        "month": int(pickup_datetime.month),
        "is_weekend": int(pickup_datetime.dayofweek >= 5),
        "distance_km": haversine_distance_km(
            pickup_latitude,
            pickup_longitude,
            dropoff_latitude,
            dropoff_longitude,
        ),
        "store_and_fwd_flag": encode_store_and_fwd_flag(data["store_and_fwd_flag"]),
    }

    return pd.DataFrame([row], columns=FEATURES)
