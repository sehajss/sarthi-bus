import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "live" / "vehicle_positions.db"

with sqlite3.connect(DB_PATH) as conn:
    rows = conn.execute("""
        SELECT
            datetime(fetched_at, 'unixepoch', 'localtime') AS fetched_time,
            route_id,
            vehicle_id,
            ROUND(lat, 5) AS latitude,
            ROUND(lon, 5) AS longitude,
            ROUND(speed, 1) AS speed
        FROM positions
        ORDER BY fetched_at DESC
        LIMIT 20
    """).fetchall()

for row in rows:
    print(row)

