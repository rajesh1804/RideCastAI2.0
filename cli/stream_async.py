import pandas as pd
import asyncio
from model.river_models import RideModel
from datetime import datetime
from utils.latency_tracker import LatencyTracker

# Config
CSV_PATH = "data/rides.csv"
STREAM_INTERVAL = 1.0  # seconds per ride

async def stream_rides(csv_path, interval):
    df = pd.read_csv(csv_path)
    feature_cols = ['pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'distance_km', 'traffic_level']
    target_cols = ['fare_amount', 'duration_min']

    latency = LatencyTracker()
    model = RideModel()

    for i, row in df.iterrows():
        x = row[feature_cols].to_dict()
        y = row[target_cols].to_dict()

        # preds = model.predict(x)
        # Measure prediction latency
        (preds, elapsed) = latency.track(model.predict, x)
        model.update(x, y)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Ride {row['ride_id']}:")
        print(f"  💰 Fare: actual ₹{y['fare_amount']} | pred ₹{round(preds['fare_pred'],2)} | MAE: {round(model.fare_mae.get(),2)}")
        print(f"  🕒 ETA: actual {y['duration_min']} min | pred {round(preds['eta_pred'],2)} | MAE: {round(model.eta_mae.get(),2)}")
        print(f"  ⚡ Latency: {round(elapsed * 1000, 2)} ms")
        print("—" * 60)

        await asyncio.sleep(interval)
    
    # Print final summary
    print("\n✅ Latency Summary:", latency.summary())

if __name__ == "__main__":
    asyncio.run(stream_rides(CSV_PATH, STREAM_INTERVAL))
