import os
import sys
import json
import pandas as pd
import argparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.river_models import RideModel, CACHE_STATS
from utils.latency_tracker import LatencyTracker

def run_batch_prediction(csv_path="data/rides.csv", max_samples=100, output_dir="cli_outputs"):
    df = pd.read_csv(csv_path)
    df = df.head(max_samples)

    feature_cols = ['pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'distance_km', 'traffic_level']
    target_cols = ['fare_amount', 'duration_min']

    model = RideModel()
    latency = LatencyTracker()

    logs = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running batch prediction"):
        x = row[feature_cols].to_dict()
        y = row[target_cols].to_dict()

        preds, elapsed = latency.track(model.predict, x)
        model.update(x, y)

        logs.append({
            "ride_id": row["ride_id"] if "ride_id" in row else idx,
            "actual_fare": y["fare_amount"],
            "pred_fare": preds["fare_pred"],
            "actual_eta": y["duration_min"],
            "pred_eta": preds["eta_pred"],
            "latency_ms": round(elapsed * 1000, 2)
        })

    # Metrics Summary
    fare_rmse = model.metrics.fare_rmse.get()
    eta_rmse = model.metrics.eta_rmse.get()
    latency_stats = latency.summary()

    fare_errors, eta_errors = model.get_top_errors()

    output = {
        "summary": {
            "total_rides": len(df),
            "fare_rmse": round(fare_rmse, 2),
            "eta_rmse": round(eta_rmse, 2),
            "latency_ms": latency_stats
        },
        "fare_top_errors": fare_errors,
        "eta_top_errors": eta_errors,
        "drift": {
            "fare_input_scores": list(model.drift_monitor.fare_input_scores),
            "fare_output_flags": list(map(bool, model.drift_monitor.fare_output_flags)),
            "eta_input_scores": list(model.drift_monitor.eta_input_scores),
            "eta_output_flags": list(map(bool, model.drift_monitor.eta_output_flags))
        },
        "logs": logs
    }

    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "predictions.csv")
    json_file = os.path.join(output_dir, "summary.json")

    pd.DataFrame(logs).to_csv(csv_file, index=False)
    with open(json_file, "w") as f:
        json.dump(output, f, indent=2)

    print("\n✅ Batch prediction completed")
    print(f"📊 Fare RMSE: {fare_rmse:.2f} | ETA RMSE: {eta_rmse:.2f}")
    print(f"⚡ Latency (ms): {latency_stats}")
    print(f"📝 Outputs written to: {csv_file}, {json_file}")

    print("✅ Cache Stats:")
    print(f"Fare:  Hits = {CACHE_STATS['fare_hits']}, Misses = {CACHE_STATS['fare_misses']}")
    print(f"ETA:   Hits = {CACHE_STATS['eta_hits']}, Misses = {CACHE_STATS['eta_misses']}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference using RideCastAI 2.0")
    parser.add_argument("--csv", type=str, default="data/rides.csv", help="Path to input rides CSV")
    parser.add_argument("--samples", type=int, default=100, help="Max number of samples to process")
    parser.add_argument("--out", type=str, default="data/cli_outputs", help="Output directory")

    args = parser.parse_args()
    run_batch_prediction(args.csv, args.samples, args.out)
