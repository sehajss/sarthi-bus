import sqlite3
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "data" / "live" / "vehicle_positions.db"
STATIC_DIR = BASE / "data" / "static"

with sqlite3.connect(DB_PATH) as conn:
    live_trips = pd.read_sql("SELECT DISTINCT trip_id, route_id FROM positions LIMIT 10", conn)
print("LIVE trip_ids sample:")
print(live_trips)

sched_files = list(STATIC_DIR.glob("schedule_*.csv"))
print("\nschedule files found:", [f.name for f in sched_files])
if sched_files:
    sched = pd.read_csv(sched_files[0])
    print("\nSTATIC trip_ids sample:")
    print(sched["trip_id"].unique()[:10])