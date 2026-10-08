# Sarthi — Weather-Aware Agentic Transit Copilot for Delhi's Bus Network

A live, agentic trip-planning and bus-tracking system built on real Delhi Transport
Stack (GTFS/GTFS-RT) data — combining a real-time data pipeline, a trained travel-time
model, and an LLM agent with tool-calling, behind a live-updating dashboard.

Built as a personal project to explore agentic AI + real operational (not toy) data,
motivated by prepping for ML/AI engineering roles in the mobility/location space.

---

## What it does

- **Live trip planner** — pick two stops, get an estimated travel time and the real
  bus route that connects them, pulled from actually-observed trips (not a static
  timetable).
- **Live bus tracking** — watch buses move on a map in near-real-time (15s refresh),
  see which are approaching or have just left a given stop.
- **Location search by name** — type a place name (e.g. "Connaught Place"), geocoded
  via OpenStreetMap, matched to the nearest real DTC stop.
- **Conversational agent ("Ask Sarthi")** — natural-language questions like *"is the
  Sarita Vihar bus running late?"* are resolved end-to-end: place name → stop →
  serving routes → live positions / predicted travel time, with no route/stop IDs
  ever exposed to the user.
- **Multi-hop routing** — for stops too far apart for a single predicted hop, the
  agent stitches together a path across multiple observed hops (graph search over
  logged data, via NetworkX) rather than refusing outright.

## Architecture

```
Delhi Transport Stack (otd.delhi.gov.in)
        │  GTFS-RT (live positions)         Open-Meteo (free, no key)
        ▼                                            ▼
  log_gtfs_rt.py  ──────────────┐        log_weather.py
        │  polls every 30s      │              │  polls every 15min
        ▼                       │              ▼
   SQLite: positions table ◄────┴──────► SQLite: weather table
        │
        ▼
  compute_delays.py  (nearest-stop matching via BallTree — see "Key decisions")
        │
        ▼
  build_travel_times.py  (derives stop-to-stop travel time labels per vehicle)
        │
        ▼
  train_model.py  (LightGBM: hour, day-of-week, route, stops, weather → travel_seconds)
        │
        ▼
  agent/agent.py  (LangGraph agent, Groq-hosted LLM, tool-calling)
        │
        ▼
  app.py  (Streamlit dashboard — live map, trip planner, chat, auto-refresh)
```

## Key decisions (and why)

**Static GTFS trip IDs don't match the live RT feed's IDs.** The official
otd.delhi.gov.in static download was unreliable, and a working third-party static
mirror uses a different ID scheme than the live feed. Rather than block on this,
delay computation was redesigned around **nearest-stop matching by GPS distance**
(via a `stops.txt`-based BallTree) instead of trip-ID joins — sidestepping the
mismatch entirely.

**The model predicts a single adjacent-stop hop, not a full journey.** Early
versions let the agent misapply a one-hop prediction to a multi-kilometer trip,
producing a confidently wrong number. This was fixed at the tool level: `predict_delay`
computes the distance between the two given stops and refuses (with an explanation)
if it's beyond ~1km, rather than extrapolating.

**Multi-hop journeys are answered by graph search, not by the LLM guessing.**
`find_path_between_stops` builds a directed graph from every observed stop-to-stop
hop in the logged data and runs Dijkstra's algorithm (via NetworkX) to find a real,
data-backed path — including which route to take per leg.

**The agent is instructed to fail honestly.** When a place name isn't found, a route
isn't in training data, or no path exists yet in the logged data, the agent is
prompted to say so plainly rather than inventing a plausible-sounding number.

## Known limitations

- **Weather signal is unvalidated.** The logging period saw almost zero rain, so
  while `precipitation_mm` and `visibility_m` are wired into the model as features,
  their real predictive value hasn't been observed yet. Feature importance during
  training showed weather mattering less than route/stop identity, which is expected
  given the lack of rain — this needs re-evaluation once rain data is collected.
- **Geographic coverage is Delhi DTC only.** Areas outside Delhi (e.g. Noida,
  Gurugram) will have no real nearby stop, and the app is designed to say so rather
  than point at a distant stop as if it were nearby.
- **Multi-hop pathfinding is data-limited.** Many stop pairs — especially ones
  further apart — won't have a connected path yet simply because the logged data
  doesn't span enough hours/routes. This improves automatically the longer the
  loggers run.
- **The RT feed has no bearing/heading field.** "Approaching vs. departed" status is
  inferred by comparing a bus's distance to a stop across two points in time, not
  from a true direction reading.

## Tech stack

Python · LangGraph / LangChain · Groq API (`openai/gpt-oss-120b`) · LightGBM ·
NetworkX · scikit-learn (BallTree) · Streamlit · pydeck · SQLite · GTFS/GTFS-RT ·
OpenStreetMap Nominatim (geocoding)

## Running it locally

```bash
# 1. Start the two background loggers (leave running)
python ingest/log_gtfs_rt.py
python ingest/log_weather.py

# 2. Once enough data has accumulated, build the training set and train
python model/compute_delays.py
python model/build_travel_times.py
python model/train_model.py

# 3. Set your free Groq API key (console.groq.com)
export GROQ_API_KEY=your_key_here

# 4. Run the app
streamlit run app.py
```

## Future work

- Validate and, if needed, re-weight the weather feature once rain data exists
- Persist category encoders' full route/stop metadata for richer agent responses
- Replace the distance/speed-based ETA fallback with a proper per-leg model chain
- Deploy the live loggers on an always-on host so the public demo reflects real-time
  data rather than a frozen snapshot

## Note

- You mjust download the GTFS static data separately, stop_times.txt was too large for the repository.