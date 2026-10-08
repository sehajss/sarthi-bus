"""
Load static GTFS into pandas DataFrames + build scheduled stop-to-stop times.
pip install gtfs-kit pandas
"""
import gtfs_kit as gk

GTFS_ZIP_PATH = "../data/static/gtfs.zip"  # point to your downloaded zip

def load_feed():
    feed = gk.read_feed(GTFS_ZIP_PATH, dist_units="km")
    return feed

def summarize(feed):
    print("Routes:", len(feed.routes))
    print("Stops:", len(feed.stops))
    print("Trips:", len(feed.trips))
    print(feed.routes[["route_id", "route_short_name", "route_long_name"]].head(10))

def scheduled_times_for_route(feed, route_id):
    """Return stop_times for a given route_id, sorted by trip and sequence."""
    trips = feed.trips[feed.trips["route_id"] == route_id]["trip_id"]
    st = feed.stop_times[feed.stop_times["trip_id"].isin(trips)]
    return st.sort_values(["trip_id", "stop_sequence"])

if __name__ == "__main__":
    feed = load_feed()
    summarize(feed)

    # pick a few routes by name (edit these to real route_long_names you saw)
    MY_ROUTES = ["764MSTLDOWN", "102STLDOWN", "819STLDOWN"]

    for rid in feed.routes[feed.routes["route_long_name"].isin(MY_ROUTES)]["route_id"]:
        df = scheduled_times_for_route(feed, rid)
        out_path = f"../data/static/schedule_{rid}.csv"
        df.to_csv(out_path, index=False)
        print(f"route {rid}: saved {len(df)} rows -> {out_path}")