import pandas as pd
import time
from model.river_models import RideModel

# Load dataset
df = pd.read_csv("data/rides.csv")

# Drop ID/timestamp for modeling
feature_cols = ['pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'distance_km', 'traffic_level']
target_cols = ['fare_amount', 'duration_min']

model = RideModel()

for i, row in df.iterrows():
    x = row[feature_cols].to_dict()
    y = row[target_cols].to_dict()

    preds = model.predict(x)
    model.update(x, y)

    # Print progress
    print(f"Ride {row['ride_id']}:")
    print(f"  📍 Fare: actual ₹{y['fare_amount']} | pred ₹{round(preds['fare_pred'],2)} | MAE: {round(model.fare_mae.get(),2)}")
    print(f"  🚗 ETA: actual {y['duration_min']} min | pred {round(preds['eta_pred'],2)} min | MAE: {round(model.eta_mae.get(),2)}")
    print("—" * 60)

    # Simulate streaming delay
    time.sleep(0.1)
