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
st.markdown(
    "<meta name='description' content='RideCastAI 2.0: Real-time Fare & ETA Prediction System with Online Learning, Model Drift Detection, Latency Tracking, and Caching. Built by Rajesh Marudhachalam.'>",
    unsafe_allow_html=True
)


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
if st.session_state.logs:
    last = st.session_state.logs[-1]
    st.sidebar.caption(f"📌 Last Prediction:\n\n{last}")

st.sidebar.markdown("---")
st.sidebar.markdown("Check out 🚖 [RideCastAI 1.0](https://huggingface.co/spaces/rajesh1804/RideCastAI).", unsafe_allow_html=True)

st.markdown("""
# 🛰️ RideCastAI 2.0  
#### Real-Time Fare & ETA Prediction with Self-Healing Drift Detection 
""")


tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📘 Overview",
    "🔮 Live Prediction", 
    "📈 Model Drift & Metrics", 
    "⚡ Latency & Cache", 
    "🛠️ Settings"
])

with tab0:
    st.title("📘 RideCastAI 2.0 Overview")
    st.markdown("""
    RideCastAI 2.0 is a real-time ML system for fare and ETA prediction in ride-hailing platforms.

    - 🔄 **Online Learning**: Model updates after every prediction.
    - 📉 **Dual Drift Detection**: Tracks input anomalies and error spikes.
    - ⚡ **ONNX + Caching Ready**: Optimized for fast, duplicate-tolerant inference.
    - 🧪 **Stress-Test Mode**: Injects drift to test self-healing capabilities.
    - 📈 **Developer Observability**: Live RMSE, drift, cache stats and latency graphs.

    Originally built as a static ML pipeline [RideCastAI 1.0](https://huggingface.co/spaces/rajesh1804/RideCastAI), this new version mimics a real production ML system under noisy, ever-changing data.
    """)

    with st.expander("📉 Architecture Diagram"):
        st.image("assets/RideCastAI2.0-architecture.png", caption="RideCastAI 2.0 Architecture")


with tab1:
    st.header("🚕 Simulate a Ride")
    st.info("📘 This tab simulates a ride and shows predicted vs actual fare & ETA.")

    with st.expander("ℹ️ How to Use This App", expanded=False):
        st.markdown("""
        1. Use the slider to simulate a ride.
        2. Watch how predictions adapt over time.
        3. Observe latency, drift, and caching on the other tabs.
        """)


    idx = st.slider("Simulated Ride Index", 0, len(df) - 1, 0)
    if idx == 0:
        st.toast("👋 Move the slider to simulate a ride. Watch the model adapt in real time!", icon="🚗")

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

    st.subheader("📉 Input Drift & Output Drift")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    plot_drift_with_overlay(ax[0], model.drift_monitor.fare_input_scores, model.drift_monitor.fare_output_flags, label="Fare", color="orange")
    plot_drift_with_overlay(ax[1], model.drift_monitor.eta_input_scores, model.drift_monitor.eta_output_flags, label="ETA", color="blue")
    st.pyplot(fig)

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

    col6, col7 = st.columns(2)
    fare_eff = CACHE_STATS['fare_hits'] / max(1, CACHE_STATS['fare_hits'] + CACHE_STATS['fare_misses']) * 100
    eta_eff = CACHE_STATS['eta_hits'] / max(1, CACHE_STATS['eta_hits'] + CACHE_STATS['eta_misses']) * 100

    eff_emoji = lambda eff: "⚡" if eff > 90 else "🔄" if eff > 50 else "🐢"
    with col6:
        st.metric(f"Fare Cache Efficiency", f"{fare_eff:.1f}% {eff_emoji(fare_eff)}", help="Higher = more cache reuse", label_visibility="visible")
    with col7:
        st.metric("ETA Cache Efficiency", f"{eta_eff:.1f}% {eff_emoji(fare_eff)}", help="Higher = more cache reuse", label_visibility="visible")


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
    col8, col9 = st.columns(2)
    with col8:
        st.caption("Toggle drift or online updates.")
        st.session_state.model.inject_drift = st.checkbox("🧪 Inject Drift (Demo Only)", value=model.inject_drift)
        st.session_state.model.enable_online_update = st.checkbox("🔄 Enable Online Updates", value=model.enable_online_update)
    with col9:
        st.caption("Resetting model will clear all logs and metrics.")
        if st.button("🔁 Reset Model (Dev Only)"):        
            st.session_state.model = RideModel()
            st.session_state.logs = []
            st.success("Model and logs reset.")

st.markdown("---")
st.markdown("""
<div style='text-align: right; font-size: 14px;'>
Built by <b><a href='https://www.linkedin.com/in/rajesh1804'>Rajesh Marudhachalam</a></b> · <a href='https://github.com/rajesh1804'>GitHub</a> · <a href='https://rajesh1804.medium.com'>Medium</a>
</div>
""", unsafe_allow_html=True)