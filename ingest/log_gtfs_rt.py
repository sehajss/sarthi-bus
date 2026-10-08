"""
Poll GTFS-RT feed and log vehicle positions to SQLite.
Run continuously: python log_gtfs_rt.py
"""
import time
import sqlite3
import requests
from google.transit import gtfs_realtime_pb2  # pip install gtfs-realtime-bindings

API_URL = "https://otd.delhi.gov.in/api/realtime/VehiclePositions.pb?key=DlPHY10IhbSsV40AIWoaBTsEBrN5iyfI"
POLL_SECONDS = 30
DB_PATH = "../data/live/vehicle_positions.db"
ROUTE_FILTER = None  # e.g. {"764", "212"} to limit; None = log all

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            fetched_at INTEGER,
            trip_id TEXT,
            route_id TEXT,
            vehicle_id TEXT,
            lat REAL,
            lon REAL,
            speed REAL,
            vehicle_timestamp INTEGER
        )
    """)
    conn.commit()
    return conn

def fetch_and_log(conn):
    resp = requests.get(API_URL, timeout=15)
    resp.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)

    fetched_at = int(time.time())
    rows = []
    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue
        v = entity.vehicle
        route_id = v.trip.route_id
        if ROUTE_FILTER and route_id not in ROUTE_FILTER:
            continue
        rows.append((
            fetched_at,
            v.trip.trip_id,
            route_id,
            v.vehicle.id,
            v.position.latitude,
            v.position.longitude,
            v.position.speed,
            v.timestamp,
        ))

    conn.executemany(
        "INSERT INTO positions VALUES (?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    print(f"[{fetched_at}] logged {len(rows)} rows")

def main():
    conn = init_db()
    while True:
        try:
            fetch_and_log(conn)
        except Exception as e:
            print("error:", e)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()