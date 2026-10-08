"""
Turn stop-visit events (from compute_delays.py) into travel-time labels:
for each vehicle, time between arriving at one stop and the next different stop.
Run after compute_delays.py has produced delays.csv.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
IN_CSV = BASE / "data" / "live" / "delays.csv"
OUT_CSV = BASE / "data" / "live" / "travel_times.csv"

def build_travel_times(df, min_seconds=30, max_seconds=3600):
    df["fetched_dt"] = pd.to_datetime(df["fetched_dt"])
    df = df.sort_values(["vehicle_id", "fetched_dt"])

    rows = []
    for vehicle_id, g in df.groupby("vehicle_id"):
        prev_stop, prev_time = None, None
        for _, r in g.iterrows():
            if prev_stop is not None and r["stop_id"] != prev_stop:
                secs = (r["fetched_dt"] - prev_time).total_seconds()
                if min_seconds <= secs <= max_seconds:  # drop noise/gaps
                    rows.append({
                        "vehicle_id": vehicle_id,
                        "route_id": r["route_id"],
                        "from_stop": prev_stop,
                        "to_stop": r["stop_id"],
                        "start_time": prev_time,
                        "end_time": r["fetched_dt"],
                        "travel_seconds": secs,
                        "hour": prev_time.hour,
                        "dayofweek": prev_time.dayofweek,
                    })
            prev_stop, prev_time = r["stop_id"], r["fetched_dt"]
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = pd.read_csv(IN_CSV)
    travel = build_travel_times(df)
    travel.to_csv(OUT_CSV, index=False)
    print(f"built {len(travel)} travel-time samples -> {OUT_CSV}")
    if not travel.empty:
        print(travel["travel_seconds"].describe())