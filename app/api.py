from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.preprocessing import prepare_features

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
XGB_MODEL_PATH = MODELS_DIR / "xgb_model.pkl"
LGBM_MODEL_PATH = MODELS_DIR / "lgbm_model.pkl"

app = FastAPI(
    title="NYC Taxi Trip Duration Prediction API",
    description="API для предсказания длительности поездки такси в Нью-Йорке.",
    version="1.0.0",
)

xgb_model = None
lgbm_model = None


class TripRequest(BaseModel):
    vendor_id: int = Field(1, ge=1, description="ID перевозчика, обычно 1 или 2")
    pickup_datetime: str = Field(
        "2016-03-15 14:30:00",
        description="Дата и время начала поездки в формате YYYY-MM-DD HH:MM:SS",
    )
    passenger_count: int = Field(1, ge=0, le=9, description="Количество пассажиров")
    pickup_longitude: float = Field(-73.985428, description="Долгота точки посадки")
    pickup_latitude: float = Field(40.748817, description="Широта точки посадки")
    dropoff_longitude: float = Field(-73.985130, description="Долгота точки высадки")
    dropoff_latitude: float = Field(40.758896, description="Широта точки высадки")
    store_and_fwd_flag: Literal["N", "Y"] = Field(
        "N",
        description="Флаг store_and_fwd_flag: N или Y",
    )


class TripPrediction(BaseModel):
    predicted_trip_duration_seconds: int
    predicted_trip_duration_minutes: float


@app.on_event("startup")
def load_models() -> None:
    global xgb_model, lgbm_model

    if not XGB_MODEL_PATH.exists() or not LGBM_MODEL_PATH.exists():
        return

    xgb_model = joblib.load(XGB_MODEL_PATH)
    lgbm_model = joblib.load(LGBM_MODEL_PATH)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "NYC Taxi Trip Duration Prediction API",
        "docs": "Open /docs to test the API",
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "xgb_model_loaded": xgb_model is not None,
        "lgbm_model_loaded": lgbm_model is not None,
        "xgb_model_path": str(XGB_MODEL_PATH),
        "lgbm_model_path": str(LGBM_MODEL_PATH),
    }


@app.post("/predict", response_model=TripPrediction)
def predict_trip_duration(request: TripRequest) -> TripPrediction:
    if xgb_model is None or lgbm_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Models are not loaded. Put xgb_model.pkl and lgbm_model.pkl "
                "into app/models or run scripts/train_and_save_models.py first."
            ),
        )

    features = prepare_features(request.model_dump())

    xgb_log_prediction = float(xgb_model.predict(features)[0])
    lgbm_log_prediction = float(lgbm_model.predict(features)[0])

    ensemble_log_prediction = (xgb_log_prediction + lgbm_log_prediction) / 2
    seconds = max(0, int(round(np.expm1(ensemble_log_prediction))))
    minutes = round(seconds / 60, 2)

    return TripPrediction(
        predicted_trip_duration_seconds=seconds,
        predicted_trip_duration_minutes=minutes,
    )
