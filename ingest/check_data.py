import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "live" / "vehicle_positions.db"

with sqlite3.connect(DB_PATH) as conn:
    pos_count = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    pos_span = conn.execute(
        "SELECT MIN(fetched_at), MAX(fetched_at) FROM positions"
    ).fetchone()
    routes = conn.execute("SELECT COUNT(DISTINCT route_id) FROM positions").fetchone()[0]

    weather_count = conn.execute("SELECT COUNT(*) FROM weather").fetchone()[0]
    weather_variety = conn.execute(
        "SELECT MIN(precipitation_mm), MAX(precipitation_mm), MIN(visibility_m), MAX(visibility_m) FROM weather"
    ).fetchone()

hours_spanned = (pos_span[1] - pos_span[0]) / 3600 if pos_span[0] else 0

print(f"positions logged: {pos_count} rows over {hours_spanned:.1f} hours, {routes} distinct routes")
print(f"weather logged: {weather_count} rows")
print(f"precipitation range: {weather_variety[0]}-{weather_variety[1]} mm")
print(f"visibility range: {weather_variety[2]}-{weather_variety[3]} m")