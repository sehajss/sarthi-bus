"""
Match logged live positions to the nearest scheduled stop (by trip_id + distance),
then compute delay = actual_time - scheduled_time.

Run after you have some positions logged + schedule_<route_id>.csv files from load_static.py.
"""
import sqlite3
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "data" / "live" / "vehicle_positions.db"
STATIC_DIR = BASE / "data" / "static"
OUT_CSV = BASE / "data" / "live" / "delays.csv"

def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

def gtfs_time_to_dt(service_date, hms):
    """GTFS times can exceed 24:00:00 (past-midnight trips)."""
    h, m, s = map(int, hms.split(":"))
    return datetime.combine(service_date, datetime.min.time()) + timedelta(hours=h, minutes=m, seconds=s)

def load_positions():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM positions", conn)
    df["fetched_dt"] = pd.to_datetime(df["fetched_at"], unit="s")
    return df

def load_stops():
    return pd.read_csv(STATIC_DIR / "stops.txt")[["stop_id", "stop_lat", "stop_lon"]]

import numpy as np
from sklearn.neighbors import BallTree

def compute_delays(positions, stops, max_match_dist_m=100):
    """
    Vectorized nearest-stop lookup (needed - looping row-by-row over millions
    of positions against thousands of stops is too slow).
    Flags vehicle-near-stop events as a dwell/congestion proxy signal,
    since we don't have matching static schedules to compute true delay yet.
    """
    stop_coords = np.radians(stops[["stop_lat", "stop_lon"]].values)
    tree = BallTree(stop_coords, metric="haversine")

    pos_coords = np.radians(positions[["lat", "lon"]].values)
    dist_rad, idx = tree.query(pos_coords, k=1)
    dist_m = dist_rad[:, 0] * 6371000  # earth radius in meters

    positions = positions.copy()
    positions["nearest_stop_idx"] = idx[:, 0]
    positions["match_dist_m"] = dist_m
    matched = positions[positions["match_dist_m"] <= max_match_dist_m].copy()
    matched["stop_id"] = stops.iloc[matched["nearest_stop_idx"]]["stop_id"].values

    return matched[["trip_id", "route_id", "vehicle_id", "stop_id", "fetched_dt", "speed", "match_dist_m"]]

if __name__ == "__main__":
    positions = load_positions()
    stops = load_stops()
    matched = compute_delays(positions, stops)
    matched.to_csv(OUT_CSV, index=False)
    print(f"matched {len(matched)} of {len(positions)} positions to a stop -> {OUT_CSV}")
    if not matched.empty:
        print(matched["speed"].describe())