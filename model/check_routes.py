import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "travel_time_model.pkl"

bundle = joblib.load(MODEL_PATH)
route_ids = list(bundle["encoders"]["route_id"].keys())
print(f"{len(route_ids)} routes in training data")
print(route_ids[:20])