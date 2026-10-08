"""
1. Join travel_times.csv with the nearest weather reading (by timestamp).
2. Train a LightGBM model predicting travel_seconds from time/route/weather features.
pip install lightgbm scikit-learn
"""
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import joblib

BASE = Path(__file__).resolve().parent.parent
TRAVEL_CSV = BASE / "data" / "live" / "travel_times.csv"
DB_PATH = BASE / "data" / "live" / "vehicle_positions.db"
MODEL_PATH = BASE / "model" / "travel_time_model.pkl"

def load_weather():
    with sqlite3.connect(DB_PATH) as conn:
        w = pd.read_sql("SELECT * FROM weather", conn)
    w["fetched_dt"] = pd.to_datetime(w["fetched_at"], unit="s")
    return w.sort_values("fetched_dt")

def join_weather(travel, weather):
    travel["start_time"] = pd.to_datetime(travel["start_time"]).astype("datetime64[ns]")
    weather["fetched_dt"] = weather["fetched_dt"].astype("datetime64[ns]")
    travel = travel.sort_values("start_time")
    merged = pd.merge_asof(
        travel, weather, left_on="start_time", right_on="fetched_dt", direction="nearest"
    )
    return merged

def train(df):
    route_cat = df["route_id"].astype("category")
    from_cat = df["from_stop"].astype("category")
    to_cat = df["to_stop"].astype("category")

    df["route_id"] = route_cat.cat.codes
    df["from_stop"] = from_cat.cat.codes
    df["to_stop"] = to_cat.cat.codes

    encoders = {
        "route_id": dict(zip(route_cat.cat.categories, range(len(route_cat.cat.categories)))),
        "from_stop": dict(zip(from_cat.cat.categories, range(len(from_cat.cat.categories)))),
        "to_stop": dict(zip(to_cat.cat.categories, range(len(to_cat.cat.categories)))),
    }

    features = ["hour", "dayofweek", "route_id", "from_stop", "to_stop",
                "temperature_c", "precipitation_mm", "visibility_m"]
    X = df[features].fillna(0)
    y = df["travel_seconds"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"MAE: {mae:.1f} seconds")
    print("Feature importance:", dict(zip(features, model.feature_importances_)))

    joblib.dump({"model": model, "features": features, "encoders": encoders}, MODEL_PATH)
    print(f"saved model -> {MODEL_PATH}")

if __name__ == "__main__":
    travel = pd.read_csv(TRAVEL_CSV)
    weather = load_weather()
    df = join_weather(travel, weather)
    train(df)