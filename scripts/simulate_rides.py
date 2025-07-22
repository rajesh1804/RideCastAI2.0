import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

N = 1000
start_time = datetime(2025, 1, 1, 6, 0, 0)

# Simulated city bounds (e.g., Bangalore)
LAT_MIN, LAT_MAX = 12.90, 13.05
LON_MIN, LON_MAX = 77.50, 77.70

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))

rides = []
for i in range(N):
    pickup_lat = np.random.uniform(LAT_MIN, LAT_MAX)
    pickup_lon = np.random.uniform(LON_MIN, LON_MAX)
    dropoff_lat = np.random.uniform(LAT_MIN, LAT_MAX)
    dropoff_lon = np.random.uniform(LON_MIN, LON_MAX)

    distance = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    traffic_level = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
    duration = round(distance * (1.5 + traffic_level * 0.5 + np.random.rand()), 2)
    fare = round(30 + distance * 10 + traffic_level * 15 + np.random.randn() * 5, 2)

    timestamp = start_time + timedelta(minutes=5*i)

    rides.append({
        "ride_id": i,
        "timestamp": timestamp.isoformat(),
        "pickup_lat": round(pickup_lat, 6),
        "pickup_lon": round(pickup_lon, 6),
        "dropoff_lat": round(dropoff_lat, 6),
        "dropoff_lon": round(dropoff_lon, 6),
        "distance_km": round(distance, 2),
        "duration_min": duration,
        "fare_amount": fare,
        "traffic_level": traffic_level
    })

df = pd.DataFrame(rides)
os.makedirs("data", exist_ok=True)
df.to_csv("data/rides.csv", index=False)
print("✅ Simulated 1000 rides → saved to data/rides.csv")
