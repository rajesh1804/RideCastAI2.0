import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.river_models import RideModel
from utils.latency_tracker import LatencyTracker
from utils.drift_plot_utils import plot_drift_with_overlay

st.set_page_config(page_title="RideCastAI 2.0", layout="wide")

# Load sample data
df = pd.read_csv("data/rides.csv")
feature_cols = ['pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'distance_km', 'traffic_level']
target_cols = ['fare_amount', 'duration_min']

# Initialize model + latency tracker
if "model" not in st.session_state:
    st.session_state.model = RideModel()
    st.session_state.latency = LatencyTracker()
    st.session_state.logs = []

model = st.session_state.model
latency = st.session_state.latency

# ---- Sidebar Logs ----
st.sidebar.title("🧾 System Logs")
for log in st.session_state.logs[-15:]:
    st.sidebar.text(log)

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
        preds, elapsed = latency.track(model.predict, x)
        model.update(x, y)

    log_msg = f"[#{model.total_updates}] Actual: ₹{y['fare_amount']} ({y['duration_min']} min) | Pred: ₹{preds['fare_pred']:.2f} ({preds['eta_pred']:.2f} min)"
    st.session_state.logs.append(log_msg)

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

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fare RMSE", round(model.fare_rmse.get(), 2))
    with col2:
        st.metric("ETA RMSE", round(model.eta_rmse.get(), 2))

    st.subheader("📉 Input Drift Scores + Output Drift Flags")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))

    plot_drift_with_overlay(ax[0], model.fare_drift_scores, model.fare_drift_output_flags, label="Fare", color="orange")
    plot_drift_with_overlay(ax[1], model.eta_drift_scores, model.eta_drift_output_flags, label="ETA", color="blue")
    st.pyplot(fig)

    st.subheader("📊 Rolling RMSE vs Baseline")
    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 4))
    ax2[0].plot(model.fare_rmse_history, label="Model RMSE", color='green')
    ax2[0].plot(model.fare_baseline_rmse, label="Baseline RMSE", color='gray', linestyle="--")
    ax2[0].set_title("Fare RMSE Comparison")
    ax2[0].legend()

    ax2[1].plot(model.eta_rmse_history, label="Model RMSE", color='purple')
    ax2[1].plot(model.eta_baseline_rmse, label="Baseline RMSE", color='gray', linestyle="--")
    ax2[1].set_title("ETA RMSE Comparison")
    ax2[1].legend()
    st.pyplot(fig2)

    st.subheader("📌 Top 5 Worst Predictions")
    fare_errors, eta_errors = model.get_top_errors()
    
    st.markdown("**Fare Errors:**")
    for (err, x, true, pred) in fare_errors:
        st.markdown(f"- Error: ₹{err:.2f} | True: ₹{true} | Pred: ₹{pred:.2f} | Features: `{x}`")

    st.markdown("**ETA Errors:**")
    for err, x, true, pred in eta_errors:
        st.markdown(f"- Error: {err:.2f} min | True: {true} | Pred: {pred:.2f} | Features: `{x}`")

    # 🔥 Top-N Error Bar Charts
    st.subheader("📉 Visual Top-5 Errors by Magnitude")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Fare Error Bar Chart**")
        fare_df = pd.DataFrame(fare_errors, columns=["error", "features", "true", "pred"])
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        sns.barplot(data=fare_df, y="error", x=fare_df.index, ax=ax3, palette="Oranges")
        ax3.set_ylabel("Absolute Error")
        ax3.set_title("Top-5 Fare Errors")
        st.pyplot(fig3)

    with col4:
        st.markdown("**ETA Error Bar Chart**")
        eta_df = pd.DataFrame(eta_errors, columns=["error", "features", "true", "pred"])
        fig4, ax4 = plt.subplots(figsize=(6, 3))
        sns.barplot(data=eta_df, y="error", x=eta_df.index, ax=ax4, palette="Purples")
        ax4.set_ylabel("Absolute Error")
        ax4.set_title("Top-5 ETA Errors")
        st.pyplot(fig4)

    st.subheader("🔬 Model Coefficients (LinearRegression)")
    fare_weights, eta_weights = model.get_model_weights()
    st.write("**Fare Model Weights:**")
    st.json(fare_weights)
    st.write("**ETA Model Weights:**")
    st.json(eta_weights)

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

    st.caption("Control toggles for simulation and diagnostics.")
    st.session_state.model.inject_drift = st.checkbox("🧪 Inject Drift (Demo Only)", value=model.inject_drift)
    st.session_state.model.enable_online_update = st.checkbox("🔄 Enable Online Updates", value=model.enable_online_update)

    with st.expander("📉 View Architecture Diagram"):
        st.image("https://i.imgur.com/YxZgvXx.png", caption="RideCastAI 2.0 Architecture")

    st.markdown("---")
    st.caption("Developed by Rajesh Marudhachalam — aiming for the top 0.001% of ML Engineers.")
