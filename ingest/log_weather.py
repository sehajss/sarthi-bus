"""
Poll Open-Meteo for Delhi weather and log alongside vehicle positions.
Run this in a separate terminal, in parallel with log_gtfs_rt.py.
"""
import time
import sqlite3
import requests
from pathlib import Path

LAT, LON = 28.6139, 77.2090  # Delhi
POLL_SECONDS = 900  # 15 min - weather doesn't change as fast as bus positions
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "live" / "vehicle_positions.db"

URL = (
    f"https://api.open-meteo.com/v1/forecast"
    f"?latitude={LAT}&longitude={LON}"
    f"&current=temperature_2m,precipitation,visibility,weather_code"
)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather (
            fetched_at INTEGER,
            temperature_c REAL,
            precipitation_mm REAL,
            visibility_m REAL,
            weather_code INTEGER
        )
    """)
    conn.commit()
    return conn

def fetch_and_log(conn):
    resp = requests.get(URL, timeout=15)
    resp.raise_for_status()
    cur = resp.json()["current"]
    fetched_at = int(time.time())
    conn.execute(
        "INSERT INTO weather VALUES (?,?,?,?,?)",
        (
            fetched_at,
            cur.get("temperature_2m"),
            cur.get("precipitation"),
            cur.get("visibility"),
            cur.get("weather_code"),
        ),
    )
    conn.commit()
    print(f"[{fetched_at}] weather logged:", cur)

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