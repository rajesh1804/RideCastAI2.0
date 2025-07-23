import streamlit as st
import pandas as pd
import altair as alt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.river_models import RideModel
from utils.latency_tracker import LatencyTracker

st.set_page_config(page_title="RideCastAI 2.0", layout="wide")

# Load sample data
df = pd.read_csv("data/rides.csv")
feature_cols = ['pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'distance_km', 'traffic_level']
target_cols = ['fare_amount', 'duration_min']

# Init model
model = RideModel()
latency = LatencyTracker()

# Warm-up with first 20 rows to avoid all-zero predictions
for i, row in df.iloc[:20].iterrows():
    x_warm = row[feature_cols].to_dict()
    y_warm = row[target_cols].to_dict()
    model.update(x_warm, y_warm)

# ---- UI Tabs ----
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Live Prediction", 
    "📈 Model Drift & Metrics", 
    "⚡ Latency Monitor", 
    "🛠️ Settings"
])

# ---- Tab 1: Live Prediction ----
with tab1:
    st.title("🚕 Live Fare & ETA Prediction")
    idx = st.slider("Simulated Ride Index", 0, len(df) - 1, 0)

    row = df.iloc[idx]
    x = row[feature_cols].to_dict()
    y = row[target_cols].to_dict()

    with st.spinner("Running prediction..."):
        (preds, elapsed) = latency.track(model.predict, x)
        model.update(x, y)

    if idx < 20:
        st.warning("⚠️ Model is still warming up — predictions may be less accurate for early samples.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Actual Fare (₹)", round(y["fare_amount"], 2))
        st.metric("Predicted Fare (₹)", round(preds["fare_pred"], 2))
        st.metric("Fare MAE", round(model.fare_mae.get(), 2))
    with col2:
        st.metric("Actual ETA (min)", round(y["duration_min"], 2))
        st.metric("Predicted ETA (min)", round(preds["eta_pred"], 2))
        st.metric("ETA MAE", round(model.eta_mae.get(), 2))

# ---- Tab 2: Drift & Metrics ----
with tab2:
    st.title("📈 Model Drift & Rolling Error")

    st.write("Real-time rolling metrics help you monitor model health over time.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fare RMSE", round(model.fare_rmse.get(), 2))
    with col2:
        st.metric("ETA RMSE", round(model.eta_rmse.get(), 2))

    st.markdown("### Fare Drift Score (HST)")
    fare_df = pd.DataFrame({
        "index": list(range(len(model.fare_drift_scores))),
        "drift_score": model.fare_drift_scores
    })
    st.altair_chart(
        alt.Chart(fare_df).mark_line().encode(
            x="index", y="drift_score"
        ).properties(height=200),
        use_container_width=True
    )

    st.markdown("### ETA Drift Score (HST)")
    eta_df = pd.DataFrame({
        "index": list(range(len(model.eta_drift_scores))),
        "drift_score": model.eta_drift_scores
    })
    st.altair_chart(
        alt.Chart(eta_df).mark_line().encode(
            x="index", y="drift_score"
        ).properties(height=200),
        use_container_width=True
    )

# ---- Tab 3: Latency Monitor ----
with tab3:
    st.title("⚡ Prediction Latency Insights")

    stats = latency.summary()
    st.metric("Min Latency (ms)", stats['min'])
    st.metric("Max Latency (ms)", stats['max'])
    st.metric("Avg Latency (ms)", stats['avg'])

    st.info("This measures the time taken for a single prediction from model input to output.")

# ---- Tab 4: Settings ----
with tab4:
    st.title("🛠️ Settings & Controls")

    st.caption("Adjust simulation interval, enable/disable online updates, change models.")
    st.write("🚧 Settings will be interactive in a later version.")
