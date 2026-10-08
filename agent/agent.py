"""
pip install langgraph langchain-groq joblib lightgbm
Requires a free API key from console.groq.com, set as GROQ_API_KEY env var.
"""
import os
import sqlite3
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "data" / "live" / "vehicle_positions.db"
MODEL_PATH = BASE / "model" / "travel_time_model.pkl"

_bundle = joblib.load(MODEL_PATH)
_model, _features, _encoders = _bundle["model"], _bundle["features"], _bundle["encoders"]

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "data" / "live" / "vehicle_positions.db"
STATIC_DIR = BASE / "data" / "static"
TRAVEL_CSV = BASE / "data" / "live" / "travel_times.csv"

@tool
def search_stops_by_name(query: str) -> str:
    """Find bus stop_ids matching a place/area name (e.g. 'Sarita Vihar'). Returns up to 5 matches.
    Tries exact substring match first, then falls back to fuzzy matching for close names."""
    stops = pd.read_csv(STATIC_DIR / "stops.txt")
    matches = stops[stops["stop_name"].str.contains(query, case=False, na=False)]
    if not matches.empty:
        return matches[["stop_id", "stop_name"]].head(5).to_string(index=False)

    import difflib
    names = stops["stop_name"].dropna().unique()
    close = difflib.get_close_matches(query, names, n=5, cutoff=0.5)
    if close:
        fuzzy_matches = stops[stops["stop_name"].isin(close)]
        return ("no exact match, but these are close: " +
                fuzzy_matches[["stop_id", "stop_name"]].head(5).to_string(index=False))
    return f"no stops found matching '{query}' - this area/landmark may not exist as a named stop in the Delhi DTC dataset"

@tool
def routes_serving_stop(stop_id: str) -> str:
    """Find which route_ids have been seen (in logged live data) passing through a given stop_id."""
    if not TRAVEL_CSV.exists():
        return "no travel history logged yet"
    t = pd.read_csv(TRAVEL_CSV)
    matches = t[(t["from_stop"].astype(str) == str(stop_id)) | (t["to_stop"].astype(str) == str(stop_id))]
    if matches.empty:
        return f"no routes seen passing through stop_id {stop_id} yet in logged data"
    routes = matches["route_id"].value_counts().head(5)
    return "routes seen at this stop: " + ", ".join(f"{r} ({c} times)" for r, c in routes.items())

@tool
def get_live_positions(route_id: str) -> str:
    """Get the latest logged positions for a given route_id."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT vehicle_id, lat, lon, speed, fetched_at FROM positions "
            "WHERE route_id = ? ORDER BY fetched_at DESC LIMIT 5",
            (route_id,),
        ).fetchall()
    return str(rows) if rows else "no live data for this route yet"

@tool
def get_latest_weather() -> str:
    """Get the most recent logged weather conditions."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT temperature_c, precipitation_mm, visibility_m, weather_code "
            "FROM weather ORDER BY fetched_at DESC LIMIT 1"
        ).fetchone()
    return str(row) if row else "no weather data yet"

def find_path_between_stops_data(from_stop_id: str, to_stop_id: str):
    """Plain function (not a tool) returning structured path data, reusable by the UI directly.
    Returns None with a reason string on failure: (None, reason) or (segments_list, total_seconds)."""
    if not TRAVEL_CSV.exists():
        return None, "no travel history logged yet"
    t = pd.read_csv(TRAVEL_CSV)
    if t.empty:
        return None, "no travel history logged yet"

    import networkx as nx
    edge_stats = t.groupby(["from_stop", "to_stop"]).agg(
        travel_seconds=("travel_seconds", "mean"),
        route_id=("route_id", lambda x: x.value_counts().idxmax()),
    ).reset_index()

    G = nx.DiGraph()
    for _, row in edge_stats.iterrows():
        G.add_edge(str(row["from_stop"]), str(row["to_stop"]),
                    weight=row["travel_seconds"], route_id=row["route_id"])

    from_id, to_id = str(from_stop_id), str(to_stop_id)
    if from_id not in G or to_id not in G:
        return None, "one or both stops have no logged hops yet, so no path can be found"

    try:
        path = nx.shortest_path(G, from_id, to_id, weight="weight")
    except nx.NetworkXNoPath:
        return None, f"no path found between stop {from_id} and stop {to_id} in the logged data so far"

    segments = []
    for a, b in zip(path[:-1], path[1:]):
        edge = G[a][b]
        segments.append({"from_stop": a, "to_stop": b, "route_id": edge["route_id"], "seconds": edge["weight"]})
    return segments, None

@tool
def find_path_between_stops(from_stop_id: str, to_stop_id: str) -> str:
    """Find a multi-hop path between two stops that may be far apart, by stitching together
    observed stop-to-stop hops from logged data (like a simplified transit router).
    Use this instead of predict_delay when the two stops are more than ~1km apart."""
    segments, err = find_path_between_stops_data(from_stop_id, to_stop_id)
    if segments is None:
        return err
    total_seconds = sum(s["seconds"] for s in segments)
    parts = [f"{s['from_stop']}->{s['to_stop']} via route {s['route_id']} (~{s['seconds']:.0f}s)" for s in segments]
    return (f"path found with {len(segments)} hop(s), total estimated {total_seconds:.0f} seconds "
            f"({total_seconds/60:.1f} min): " + " | ".join(parts))

@tool
def predict_delay(route_id: str, from_stop: str = "", to_stop: str = "") -> str:
    """Predict expected travel time in seconds for a SINGLE stop-to-stop hop on a route,
    given current time and weather. This is NOT valid for a full multi-stop journey -
    if from_stop and to_stop are more than ~1km apart, this tool will refuse and say so,
    since the model was only trained on adjacent-stop hops."""
    with sqlite3.connect(DB_PATH) as conn:
        w = conn.execute(
            "SELECT temperature_c, precipitation_mm, visibility_m FROM weather "
            "ORDER BY fetched_at DESC LIMIT 1"
        ).fetchone()
    if not w:
        return "no weather data available yet"
    temp, precip, vis = w
    now = datetime.now()

    if from_stop and to_stop:
        stops = pd.read_csv(STATIC_DIR / "stops.txt")
        try:
            f = stops[stops["stop_id"].astype(str) == str(from_stop)].iloc[0]
            t = stops[stops["stop_id"].astype(str) == str(to_stop)].iloc[0]
            import math
            r = 6371000
            p1, p2 = math.radians(f["stop_lat"]), math.radians(t["stop_lat"])
            dphi = math.radians(t["stop_lat"] - f["stop_lat"])
            dlmb = math.radians(t["stop_lon"] - f["stop_lon"])
            a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
            dist_km = (2 * r * math.asin(math.sqrt(a))) / 1000
            if dist_km > 1.5:
                return (f"from_stop and to_stop are {dist_km:.1f}km apart - this model only predicts a "
                        f"single adjacent-stop hop (trained on gaps under ~1km), so it cannot give a "
                        f"reliable full-journey estimate for this distance. Report this limitation "
                        f"instead of giving a number.")
        except IndexError:
            pass

    encoders_route = {str(k): v for k, v in _encoders["route_id"].items()}
    encoders_from = {str(k): v for k, v in _encoders["from_stop"].items()}
    encoders_to = {str(k): v for k, v in _encoders["to_stop"].items()}

    route_code = encoders_route.get(str(route_id), -1)
    from_code = encoders_from.get(str(from_stop), -1)
    to_code = encoders_to.get(str(to_stop), -1)
    if route_code == -1:
        return f"route_id '{route_id}' wasn't seen in training data - no reliable prediction"

    row = pd.DataFrame([{
        "hour": now.hour, "dayofweek": now.weekday(),
        "route_id": route_code, "from_stop": from_code, "to_stop": to_code,
        "temperature_c": temp, "precipitation_mm": precip, "visibility_m": vis,
    }])[_features]
    pred_seconds = _model.predict(row)[0]
    return f"predicted travel time for this single hop: {pred_seconds:.0f} seconds (temp={temp}C, precip={precip}mm, visibility={vis}m)"

tools = [search_stops_by_name, routes_serving_stop, get_live_positions, get_latest_weather, predict_delay, find_path_between_stops]
llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.environ["GROQ_API_KEY"])

SYSTEM_PROMPT = """You are a transit assistant reporting model predictions, not ground truth.

When someone asks about a place/area by name (e.g. "is the Sarita Vihar bus running late?", "which buses are near X?"):
1. Call search_stops_by_name to find matching stop_id(s) for that place.
2. Call routes_serving_stop on the best-matching stop_id to find which routes serve it.
3. Call get_live_positions and/or predict_delay using that route_id to answer the actual question.
Never ask the user for a route_id or stop_id directly - resolve it yourself using these tools first.

For a JOURNEY BETWEEN TWO STOPS:
- predict_delay only works for a SINGLE adjacent hop (under ~1km). It will refuse if the stops are farther apart.
- For anything farther apart (a real cross-city trip), use find_path_between_stops instead. It stitches
  together multiple real observed hops from logged data into a full path with total time and any bus changes needed.
- If find_path_between_stops also fails (no path in the data yet), say plainly that no route has been observed
  connecting those two stops in the data collected so far - don't invent one.

Rules:
- Never refer to yourself in the third person or narrate your own process ("I looked up...", "when I asked the system...", "the system reported..."). You ARE the system - just answer directly, e.g. "Route 1900 serves both stops, but I don't have a full path stitched together yet" not "I asked the system to stitch together a path and it reported...".
- All time estimates are based on historical/recent logged data and current weather, not live schedule ground truth.
- Never say a route "is delayed" or claim weather "caused" a number. Say "the model estimates X" instead.
- If precipitation is 0, do not attribute any prediction to rain/weather - the model showed weather has weak influence when there's no rain.
- If a tool returns an error or a limitation message, relay it plainly rather than inventing a number or route.
- If search_stops_by_name finds nothing (even after fuzzy matching), tell the user this landmark/area may not exist as a named stop in the dataset rather than guessing.
"""

agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)

if __name__ == "__main__":
    result = agent.invoke({"messages": [("user", "Is the Sarita Vihar bus running late right now?")]})
    print(result["messages"][-1].content)