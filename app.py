"""
pip install streamlit pydeck scikit-learn streamlit-autorefresh
Run: streamlit run app.py
"""
import sqlite3
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from sklearn.neighbors import BallTree
from streamlit_autorefresh import st_autorefresh

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "data" / "live" / "vehicle_positions.db"
STATIC_DIR = BASE / "data" / "static"

import sys
sys.path.append(str(BASE / "agent"))
from agent import agent, predict_delay, find_path_between_stops_data  # noqa

TRAVEL_CSV = BASE / "data" / "live" / "travel_times.csv"

@st.cache_data(ttl=300)
def load_travel_times():
    return pd.read_csv(TRAVEL_CSV)

def find_routes_between(from_stop_id, to_stop_id):
    """Look at real logged trips to find which routes have actually gone from A to B."""
    t = load_travel_times()
    match = t[(t["from_stop"].astype(str) == str(from_stop_id)) &
              (t["to_stop"].astype(str) == str(to_stop_id))]
    if match.empty:
        return None
    summary = match.groupby("route_id")["travel_seconds"].agg(["mean", "count"]).reset_index()
    return summary.sort_values("count", ascending=False)

def live_buses_on_route(route_id, window_seconds=300):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            "SELECT vehicle_id, lat, lon, fetched_at FROM positions "
            "WHERE route_id = ? AND fetched_at > (SELECT MAX(fetched_at) - ? FROM positions) "
            "ORDER BY fetched_at DESC", conn, params=(str(route_id), window_seconds)
        )
    if df.empty:
        return df
    return df.sort_values("fetched_at", ascending=False).drop_duplicates("vehicle_id")

st.set_page_config(page_title="Sarthi Bus", layout="wide", page_icon="🚌")
st_autorefresh(interval=15000, key="live_refresh")

# ---------- theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --bg: #12141C;
    --panel: #1B1E29;
    --text: #E8EAF0;
    --muted: #8A93A6;
    --amber: #F2A93B;
    --green: #3C7A5C;
    --line: #2A2E3D;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }

.board-header {
    display: flex; align-items: baseline; justify-content: space-between;
    border-bottom: 2px solid var(--amber); padding-bottom: 14px; margin-bottom: 28px;
}
.board-header h1 { font-size: 28px; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.board-header .clock { font-family: 'IBM Plex Mono', monospace; color: var(--amber); font-size: 15px; }

.section-label { color: var(--muted); font-size: 13px; margin-bottom: 8px; font-weight: 500; }

.route-tile {
    display: flex; align-items: center; gap: 16px;
    background: var(--panel); border-left: 3px solid var(--amber);
    padding: 14px 18px; margin-bottom: 10px; border-radius: 4px;
}
.route-tile .num {
    font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 20px;
    color: var(--amber); min-width: 60px;
}
.route-tile .detail { color: var(--text); font-size: 14px; line-height: 1.5; }
.route-tile .detail .muted { color: var(--muted); }

.stat-row { display: flex; gap: 18px; margin-bottom: 6px; }
.stat-box { background: var(--panel); border-radius: 4px; padding: 12px 16px; flex: 1; }
.stat-box .label { color: var(--muted); font-size: 12px; }
.stat-box .value { font-family: 'IBM Plex Mono', monospace; font-size: 22px; color: var(--text); margin-top: 4px; }

hr.divider { border: none; border-top: 1px solid var(--line); margin: 32px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="board-header">
  <h1>Sarthi &mdash; Live Bus</h1>
  <div class="clock">DELHI &middot; {datetime.now().strftime('%H:%M')}</div>
</div>
""", unsafe_allow_html=True)

# ---------- data loaders ----------
@st.cache_data(ttl=60)
def load_stops():
    return pd.read_csv(STATIC_DIR / "stops.txt")

@st.cache_resource
def stop_tree(stops):
    coords = np.radians(stops[["stop_lat", "stop_lon"]].values)
    return BallTree(coords, metric="haversine")

def latest_positions():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(
            "SELECT lat, lon, route_id, vehicle_id FROM positions "
            "WHERE fetched_at > (SELECT MAX(fetched_at) - 120 FROM positions)", conn
        )

def latest_weather():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM weather ORDER BY fetched_at DESC LIMIT 1", conn)

stops = load_stops()
tree = stop_tree(stops)
positions = latest_positions()
weather = latest_weather()

# ---------- route planner ----------
st.markdown('<div class="section-label">PLAN A TRIP</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([2, 2, 1])
stop_names = stops["stop_name"].dropna().unique()
with c1:
    from_name = st.selectbox("From", stop_names, key="from_stop", label_visibility="collapsed", placeholder="From stop")
with c2:
    to_name = st.selectbox("To", stop_names, key="to_stop", label_visibility="collapsed", placeholder="To stop")
with c3:
    go = st.button("Find route", use_container_width=True)

if go and from_name and to_name:
    st.session_state["planned_trip"] = {"from_name": from_name, "to_name": to_name}

if "planned_trip" in st.session_state:
    trip = st.session_state["planned_trip"]
    from_row = stops[stops["stop_name"] == trip["from_name"]].iloc[0]
    to_row = stops[stops["stop_name"] == trip["to_name"]].iloc[0]
    routes = find_routes_between(from_row["stop_id"], to_row["stop_id"])

    if st.button("Clear trip", key="clear_trip"):
        del st.session_state["planned_trip"]
        st.rerun()

    if routes is None:
        segments, err = find_path_between_stops_data(from_row["stop_id"], to_row["stop_id"])
        if segments is None:
            st.markdown(f"""
            <div class="route-tile">
              <div class="num">?</div>
              <div class="detail">No bus has been seen going directly from <b>{trip['from_name']}</b> to <b>{trip['to_name']}</b>, and no multi-bus path has been observed either yet ({err}). Try a nearby pair of stops, or check back once more data is logged.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            total_seconds = sum(s["seconds"] for s in segments)
            stop_lookup = stops.set_index(stops["stop_id"].astype(str))["stop_name"]

            st.markdown(f"""
            <div class="route-tile">
              <div class="num">{len(segments)}x</div>
              <div class="detail">
                No direct bus, but you can get from <b>{trip['from_name']}</b> to <b>{trip['to_name']}</b> with <b>{len(segments)} bus{'es' if len(segments) > 1 else ''}</b>, about <b>{total_seconds/60:.0f} min</b> total.
              </div>
            </div>
            """, unsafe_allow_html=True)

            for i, seg in enumerate(segments, 1):
                from_stop_name = stop_lookup.get(str(seg["from_stop"]), seg["from_stop"])
                to_stop_name = stop_lookup.get(str(seg["to_stop"]), seg["to_stop"])
                st.markdown(f"""
                <div class="route-tile">
                  <div class="num">{seg['route_id']}</div>
                  <div class="detail">Leg {i}: <b>{from_stop_name}</b> &rarr; <b>{to_stop_name}</b> &mdash; ~{seg['seconds']:.0f}s on route {seg['route_id']}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        top = routes.iloc[0]
        route_id, historical_avg, n_trips = top["route_id"], top["mean"], int(top["count"])
        pred = predict_delay.invoke({
            "route_id": str(route_id),
            "from_stop": str(from_row["stop_id"]),
            "to_stop": str(to_row["stop_id"]),
        })
        # pull the seconds number back out if the tool returned a clean prediction
        import re
        m = re.search(r"([\d.]+) seconds", pred)
        est_minutes = f"{float(m.group(1))/60:.0f} min" if m else f"{historical_avg/60:.0f} min"

        st.markdown(f"""
        <div class="route-tile">
          <div class="num">{route_id}</div>
          <div class="detail">
            <b>Route {route_id}</b> gets you from <b>{trip['from_name']}</b> to <b>{trip['to_name']}</b> in about <b>{est_minutes}</b> right now.
            <br><span class="muted">Based on {n_trips} past trips on this route between these stops &middot; live positions refresh every 15s</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        buses = live_buses_on_route(route_id)

        layers = [
            pdk.Layer("LineLayer",
                data=[{"from": [from_row["stop_lon"], from_row["stop_lat"]],
                       "to": [to_row["stop_lon"], to_row["stop_lat"]]}],
                get_source_position="from", get_target_position="to",
                get_color=[242, 169, 59], get_width=4),
            pdk.Layer("ScatterplotLayer",
                data=pd.DataFrame([
                    {"lat": from_row["stop_lat"], "lon": from_row["stop_lon"], "label": "From"},
                    {"lat": to_row["stop_lat"], "lon": to_row["stop_lon"], "label": "To"},
                ]),
                get_position="[lon, lat]", get_radius=30, get_fill_color=[232, 234, 240]),
        ]
        if not buses.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=buses,
                get_position="[lon, lat]", get_radius=35,
                get_fill_color=[60, 122, 92], pickable=True,
            ))

        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=(from_row["stop_lat"] + to_row["stop_lat"]) / 2,
                longitude=(from_row["stop_lon"] + to_row["stop_lon"]) / 2,
                zoom=13,
            ),
            layers=layers,
            tooltip={"text": "Bus {vehicle_id}"},
        ), height=280)

        if buses.empty:
            st.markdown('<p class="muted" style="font-size:13px;">No buses seen on this route in the last 5 minutes.</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="section-label">{len(buses)} BUS(ES) LIVE ON ROUTE {route_id} &mdash; updates every 15s</div>', unsafe_allow_html=True)
            for _, b in buses.iterrows():
                ago = int(pd.Timestamp.now().timestamp() - b["fetched_at"])
                st.markdown(f"""
                <div class="route-tile">
                  <div class="num">{b['vehicle_id']}</div>
                  <div class="detail">last seen {ago}s ago<br><span class="muted">{b['lat']:.4f}, {b['lon']:.4f}</span></div>
                </div>
                """, unsafe_allow_html=True)

def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2 * r * np.arcsin(np.sqrt(a))

@st.cache_data(ttl=3600)
def geocode_place(query):
    """Free geocoding via OpenStreetMap Nominatim - no API key needed."""
    import requests
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": f"{query}, Delhi, India", "format": "json", "limit": 1},
        headers={"User-Agent": "SarthiBusApp/1.0"}, timeout=10,
    )
    results = resp.json()
    if not results:
        return None
    return float(results[0]["lat"]), float(results[0]["lon"]), results[0].get("display_name", query)

def buses_near_stop_with_trend(stop_lat, stop_lon, window_seconds=300, radius_km=3):
    """Classify nearby buses as approaching or departed, by comparing distance-to-stop
    at the start vs end of the time window (needed since the RT feed has no bearing field)."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            "SELECT vehicle_id, route_id, lat, lon, fetched_at FROM positions "
            "WHERE fetched_at > (SELECT MAX(fetched_at) - ? FROM positions)", conn, params=(window_seconds,)
        )
    if df.empty:
        return df
    df["dist_m"] = haversine_m(df["lat"], df["lon"], stop_lat, stop_lon)
    near_vehicles = df[df["dist_m"] <= radius_km * 1000]["vehicle_id"].unique()
    df = df[df["vehicle_id"].isin(near_vehicles)].sort_values("fetched_at")

    rows = []
    for vid, g in df.groupby("vehicle_id"):
        first, last = g.iloc[0], g.iloc[-1]
        status = "approaching" if last["dist_m"] < first["dist_m"] else "departed"
        rows.append({
            "vehicle_id": vid, "route_id": last["route_id"], "lat": last["lat"], "lon": last["lon"],
            "dist_m": last["dist_m"], "status": status,
        })
    return pd.DataFrame(rows).sort_values("dist_m")

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">FIND YOUR STOP &mdash; type a location</div>', unsafe_allow_html=True)
place_query = st.text_input("Location", value="Connaught Place", label_visibility="collapsed", placeholder="e.g. Connaught Place, Lajpat Nagar, India Gate")

geo = geocode_place(place_query) if place_query else None

if place_query and not geo:
    st.markdown(f"""
    <div class="route-tile"><div class="num">?</div>
    <div class="detail">Couldn't find "<b>{place_query}</b>" &mdash; try a more specific or well-known place name.</div></div>
    """, unsafe_allow_html=True)
elif geo:
    user_lat, user_lon, matched_name = geo
    dist, idx = tree.query(np.radians([[user_lat, user_lon]]), k=5)
    nearest_m = dist[0][0] * 6371000
    nearest_stop = stops.iloc[idx[0][0]]

    st.markdown(f'<p class="muted" style="font-size:13px;">Matched: {matched_name}</p>', unsafe_allow_html=True)

    if nearest_m > 5000:
        st.markdown(f"""
        <div class="route-tile"><div class="num">!</div>
        <div class="detail">The closest stop in this dataset is <b>{nearest_m/1000:.1f} km</b> away &mdash;
        likely outside Delhi DTC coverage.</div></div>
        """, unsafe_allow_html=True)
    else:
        buses = buses_near_stop_with_trend(nearest_stop["stop_lat"], nearest_stop["stop_lon"])

        layers = [
            pdk.Layer("ScatterplotLayer", data=stops, get_position="[stop_lon, stop_lat]",
                      get_radius=15, get_fill_color=[140, 144, 156, 255]),
            pdk.Layer("ScatterplotLayer", data=pd.DataFrame([{"lat": user_lat, "lon": user_lon}]),
                      get_position="[lon, lat]", get_radius=45, get_fill_color=[232, 234, 240]),
            pdk.Layer("ScatterplotLayer", data=pd.DataFrame([{
                          "lat": nearest_stop["stop_lat"], "lon": nearest_stop["stop_lon"]}]),
                      get_position="[lon, lat]", get_radius=40, get_fill_color=[242, 169, 59]),
        ]
        if not buses.empty:
            approaching = buses[buses["status"] == "approaching"]
            departed = buses[buses["status"] == "departed"]
            if not approaching.empty:
                layers.append(pdk.Layer("ScatterplotLayer", data=approaching, get_position="[lon, lat]",
                                         get_radius=35, get_fill_color=[60, 122, 92], pickable=True))
            if not departed.empty:
                layers.append(pdk.Layer("ScatterplotLayer", data=departed, get_position="[lon, lat]",
                                         get_radius=35, get_fill_color=[150, 60, 60], pickable=True))

        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(latitude=user_lat, longitude=user_lon, zoom=14),
            layers=layers,
            tooltip={"text": "Bus {vehicle_id} - Route {route_id}"},
        ), height=420)
        st.markdown('<p class="muted" style="font-size:12px;">Gray = all bus stops &middot; white = you &middot; amber = your nearest stop &middot; green = approaching buses &middot; red = just-departed buses &middot; updates every 15s</p>', unsafe_allow_html=True)

        st.markdown(f'<div class="section-label">NEAREST STOP: {nearest_stop["stop_name"]} ({nearest_m:.0f}m away)</div>', unsafe_allow_html=True)
        if buses.empty:
            st.markdown('<p class="muted" style="font-size:13px;">No buses seen near this stop in the last 5 minutes.</p>', unsafe_allow_html=True)
        else:
            for _, b in buses.head(8).iterrows():
                st.markdown(f"""
                <div class="route-tile">
                  <div class="num">{b['route_id']}</div>
                  <div class="detail">Bus <b>{b['vehicle_id']}</b> &mdash; {b['dist_m']:.0f}m away &mdash; <b>{b['status']}</b></div>
                </div>
                """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ---------- live map + weather ----------
map_col, stat_col = st.columns([2, 1])
with map_col:
    st.markdown(f'<div class="section-label">LIVE BUSES ({len(positions)} in last 2 min)</div>', unsafe_allow_html=True)
    if not positions.empty:
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(latitude=28.6139, longitude=77.2090, zoom=10),
            layers=[pdk.Layer(
                "ScatterplotLayer", data=positions,
                get_position="[lon, lat]", get_radius=40,
                get_fill_color=[242, 169, 59], pickable=True,
            )],
            tooltip={"text": "Route {route_id}\nVehicle {vehicle_id}"},
        ), height=360)
    else:
        st.info("No recent positions — is log_gtfs_rt.py running?")

with stat_col:
    st.markdown('<div class="section-label">CONDITIONS NOW</div>', unsafe_allow_html=True)
    if not weather.empty:
        w = weather.iloc[0]
        st.markdown(f"""
        <div class="stat-row"><div class="stat-box"><div class="label">TEMP</div><div class="value">{w['temperature_c']}°C</div></div></div>
        <div class="stat-row"><div class="stat-box"><div class="label">RAIN</div><div class="value">{w['precipitation_mm']}mm</div></div></div>
        <div class="stat-row"><div class="stat-box"><div class="label">VISIBILITY</div><div class="value">{w['visibility_m']:.0f}m</div></div></div>
        """, unsafe_allow_html=True)
    else:
        st.info("No weather data yet")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ---------- chat ----------
st.markdown('<div class="section-label">ASK SARTHI</div>', unsafe_allow_html=True)
if "history" not in st.session_state:
    st.session_state.history = []
for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

user_input = st.chat_input("e.g. Is route 1 running slow right now?")
if user_input:
    st.session_state.history.append(("user", user_input))
    st.chat_message("user").write(user_input)
    with st.spinner("checking..."):
        result = agent.invoke({"messages": [("user", user_input)]})
        reply = result["messages"][-1].content
    st.session_state.history.append(("assistant", reply))
    st.chat_message("assistant").write(reply)