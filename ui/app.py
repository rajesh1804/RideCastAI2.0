import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.river_models import RideModel, CACHE_STATS, CACHE_HIT_HISTORY, reset_cache_stats
from utils.latency_tracker import LatencyTracker
from utils.drift_plot_utils import plot_drift_with_overlay

st.set_page_config(page_title="RideCastAI 2.0", layout="wide")

df = pd.read_csv("data/rides.csv")
feature_cols = ['pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'distance_km', 'traffic_level']
target_cols = ['fare_amount', 'duration_min']

if "model" not in st.session_state:
    st.session_state.model = RideModel()
    st.session_state.latency = LatencyTracker()
    st.session_state.logs = []

model = st.session_state.model
latency = st.session_state.latency

st.sidebar.title("🧾 Latest Logs")
log_df = pd.DataFrame(st.session_state.logs[-15:], columns=["Prediction Log"])
st.sidebar.dataframe(log_df, use_container_width=True, hide_index=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Live Prediction", 
    "📈 Model Drift & Metrics", 
    "⚡ Latency & Cache", 
    "🛠️ Settings"
])

with tab1:
    st.title("🚕 Live Fare & ETA Prediction")
    st.info("📘 This tab simulates a ride and shows predicted vs actual fare & ETA.")
    idx = st.slider("Simulated Ride Index", 0, len(df) - 1, 0)
    row = df.iloc[idx]
    x = row[feature_cols].to_dict()
    y = row[target_cols].to_dict()

    with st.spinner("Running prediction..."):
        preds, elapsed = latency.track(model.predict, x)
        model.update(x, y)

    st.session_state.logs.append(
        f"[#{model.total_updates}] Actual: ₹{y['fare_amount']} ({y['duration_min']} min) | Pred: ₹{preds['fare_pred']:.2f} ({preds['eta_pred']:.2f} min)"
    )

    if idx < 20:
        st.warning("⚠️ Model is still warming up — early predictions may be inaccurate.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Actual Fare (₹)", round(y["fare_amount"], 2))
        st.metric("Predicted Fare (₹)", round(preds["fare_pred"], 2))
        st.metric("Fare MAE", round(model.metrics.fare_mae.get(), 2))
    with col2:
        st.metric("Actual ETA (min)", round(y["duration_min"], 2))
        st.metric("Predicted ETA (min)", round(preds["eta_pred"], 2))
        st.metric("ETA MAE", round(model.metrics.eta_mae.get(), 2))

with tab2:
    st.title("📈 Drift, RMSE, and Errors")
    st.info("📘 This tab tracks model error, drift signals, and worst-case predictions.")

    if model.inject_drift:
        st.info("🧪 Drift Injection Active — model is being stressed with noisy targets.")

    if len(model.metrics.fare_rmse_history) >= 10:
        recent = model.metrics.fare_rmse_history[-5:]
        prev = model.metrics.fare_rmse_history[-10:-5]
        if max(recent) > 1.5 * max(prev):
            st.error("🚨 RMSE Spike Detected! Consider resetting or lowering learning rate.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fare RMSE", round(model.metrics.fare_rmse.get(), 2))
    with col2:
        st.metric("ETA RMSE", round(model.metrics.eta_rmse.get(), 2))

    st.subheader("📉 Input Drift & Output Drift")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    plot_drift_with_overlay(ax[0], model.drift_monitor.fare_input_scores, model.drift_monitor.fare_output_flags, label="Fare", color="orange")
    plot_drift_with_overlay(ax[1], model.drift_monitor.eta_input_scores, model.drift_monitor.eta_output_flags, label="ETA", color="blue")
    st.pyplot(fig)

    st.subheader("📊 RMSE vs Baseline (Step View)")
    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 4))
    ax2[0].step(range(len(model.metrics.fare_rmse_history)), model.metrics.fare_rmse_history, where="mid", label="Model RMSE", color='green')
    ax2[0].step(range(len(model.metrics.fare_baseline_rmse)), model.metrics.fare_baseline_rmse, where="mid", label="Baseline", color='gray', linestyle="--")
    ax2[0].set_title("Fare RMSE"); ax2[0].legend()

    ax2[1].step(range(len(model.metrics.eta_rmse_history)), model.metrics.eta_rmse_history, where="mid", label="Model RMSE", color='purple')
    ax2[1].step(range(len(model.metrics.eta_baseline_rmse)), model.metrics.eta_baseline_rmse, where="mid", label="Baseline", color='gray', linestyle="--")
    ax2[1].set_title("ETA RMSE"); ax2[1].legend()
    st.pyplot(fig2)

    st.subheader("📌 Top 5 Errors with % Error")
    fare_errors, eta_errors = model.get_top_errors()

    with st.expander("📊 Fare Errors (Bar Chart)"):
        fare_df = pd.DataFrame(fare_errors, columns=["error", "features", "true", "pred", "percent_error"])
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        sns.barplot(data=fare_df, y="error", x=fare_df.index, ax=ax3, palette="Oranges")
        ax3.set_ylabel("Absolute Error"); ax3.set_title("Top-5 Fare Errors")
        st.pyplot(fig3)

    with st.expander("📊 ETA Errors (Bar Chart)"):
        eta_df = pd.DataFrame(eta_errors, columns=["error", "features", "true", "pred", "percent_error"])
        fig4, ax4 = plt.subplots(figsize=(6, 3))
        sns.barplot(data=eta_df, y="error", x=eta_df.index, ax=ax4, palette="Purples")
        ax4.set_ylabel("Absolute Error"); ax4.set_title("Top-5 ETA Errors")
        st.pyplot(fig4)

    st.subheader("🔬 Model Coefficients")
    fare_weights, eta_weights = model.get_model_weights()
    with st.expander("📉 Fare Model Weights"):
        st.json(fare_weights)
    with st.expander("⏱️ ETA Model Weights"):
        st.json(eta_weights)

with tab3:
    st.title("⚡ Latency Monitor & Cache Efficiency")
    st.info("📘 Shows inference speed and cache hit/miss counts.")

    stats = latency.summary()
    col1, col2, col3 = st.columns(3)
    col1.metric("Min Latency (ms)", stats['min'])
    col2.metric("Max Latency (ms)", stats['max'])
    col3.metric("Avg Latency (ms)", stats['avg'])

    st.subheader("🗂️ Joblib Cache Stats")
    col4, col5 = st.columns(2)
    with col4:
        st.metric("Fare Cache Hits", CACHE_STATS["fare_hits"])
        st.metric("Fare Cache Misses", CACHE_STATS["fare_misses"])
    with col5:
        st.metric("ETA Cache Hits", CACHE_STATS["eta_hits"])
        st.metric("ETA Cache Misses", CACHE_STATS["eta_misses"])

    if st.button("🔄 Reset Cache Stats (Dev Only)"):
        reset_cache_stats()
        st.success("Cache stats and hit history reset.")

    st.subheader("📉 Cache Hit Ratio Over Time")
    if CACHE_HIT_HISTORY["fare"]:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        ax[0].plot(CACHE_HIT_HISTORY["fare"], label="Fare Cache Hit %", color="orange")
        ax[1].plot(CACHE_HIT_HISTORY["eta"], label="ETA Cache Hit %", color="purple")
        ax[0].set_ylim(0, 100)
        ax[1].set_ylim(0, 100)
        ax[0].set_ylabel("Hit Ratio %")
        ax[0].set_xlabel("Prediction Index")
        ax[0].legend()
        ax[1].set_ylabel("Hit Ratio %")
        ax[1].set_xlabel("Prediction Index")
        ax[1].legend()
        st.pyplot(fig)
    else:
        st.info("ℹ️ No cache hit history yet — make some predictions.")



with tab4:
    st.title("🛠️ Settings")
    st.caption("Toggle drift or online updates.")
    st.session_state.model.inject_drift = st.checkbox("🧪 Inject Drift (Demo Only)", value=model.inject_drift)
    st.session_state.model.enable_online_update = st.checkbox("🔄 Enable Online Updates", value=model.enable_online_update)

    with st.expander("📉 Architecture Diagram"):
        st.image("https://i.imgur.com/YxZgvXx.png", caption="RideCastAI 2.0 Architecture")

    st.caption("Built by Rajesh Marudhachalam.")
