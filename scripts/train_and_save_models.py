from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from app.preprocessing import FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    earth_radius_km = 6371

    dlat = np.radians(df["dropoff_latitude"] - df["pickup_latitude"])
    dlon = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(df["pickup_latitude"]))
        * np.cos(np.radians(df["dropoff_latitude"]))
        * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["distance_km"] = earth_radius_km * c

    for col in ["vendor_id", "passenger_count"]:
        df[col] = df[col].astype("category").cat.codes

    df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)

    return df


def find_train_csv() -> Path:
    candidates = [
        Path("data/train.csv"),
        Path("notebooks/train.csv"),
        Path("train.csv"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "train.csv not found. Put it into data/train.csv, notebooks/train.csv or project root."
    )


def main() -> None:
    train_path = find_train_csv()
    print(f"Reading train data from: {train_path}")

    train_df = pd.read_csv(train_path)
    train_df = add_features(train_df)

    initial_len = len(train_df)
    train_df = train_df[
        (train_df["trip_duration"] > 60)
        & (train_df["trip_duration"] < 7200)
    ]
    print(f"Removed {initial_len - len(train_df)} duration outliers.")

    train_df = train_df.drop_duplicates()

    x_full = train_df[FEATURES]
    y_full = np.log1p(train_df["trip_duration"])

    print("Training XGBoost...")
    xgb_final = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        eval_metric="rmse",
    )
    xgb_final.fit(x_full, y_full)

    print("Training LightGBM...")
    lgbm_final = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    lgbm_final.fit(x_full, y_full)

    models_dir = Path("app/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(xgb_final, models_dir / "xgb_model.pkl")
    joblib.dump(lgbm_final, models_dir / "lgbm_model.pkl")

    print("Models saved:")
    print(models_dir / "xgb_model.pkl")
    print(models_dir / "lgbm_model.pkl")


if __name__ == "__main__":
    main()
