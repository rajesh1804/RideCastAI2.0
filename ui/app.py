# import streamlit as st
# import pandas as pd

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from model.river_models import RideModel
# from utils.latency_tracker import LatencyTracker
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="RideCastAI 2.0", layout="wide")

# # Load sample data
# df = pd.read_csv("data/rides.csv")
# feature_cols = ['pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'distance_km', 'traffic_level']
# target_cols = ['fare_amount', 'duration_min']

# # Init model
# @st.cache_resource
# def get_model():
#     return RideModel()

# model = get_model()
# latency = LatencyTracker()

# # Warm-up with first 20 rows to avoid all-zero predictions
# for i, row in df.iloc[:20].iterrows():
#     x_warm = row[feature_cols].to_dict()
#     y_warm = row[target_cols].to_dict()
#     model.update(x_warm, y_warm)

# # ---- UI Tabs ----
# tab1, tab2, tab3, tab4 = st.tabs([
#     "🔮 Live Prediction", 
#     "📈 Model Drift & Metrics", 
#     "⚡ Latency Monitor", 
#     "🛠️ Settings"
# ])

# # ---- Tab 1: Live Prediction ----
# with tab1:
#     st.title("🚕 Live Fare & ETA Prediction")
#     idx = st.slider("Simulated Ride Index", 0, len(df) - 1, 0)

#     row = df.iloc[idx]
#     x = row[feature_cols].to_dict()
#     y = row[target_cols].to_dict()

#     with st.spinner("Running prediction..."):
#         (preds, elapsed) = latency.track(model.predict, x)
#         model.update(x, y)

#     if idx < 20:
#         st.warning("⚠️ Model is still warming up — predictions may be less accurate for early samples.")

#     st.text(f"✅ Model updated at index {idx}")

#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Actual Fare (₹)", round(y["fare_amount"], 2))
#         st.metric("Predicted Fare (₹)", round(preds["fare_pred"], 2))
#         st.metric("Fare MAE", round(model.fare_mae.get(), 2))
#     with col2:
#         st.metric("Actual ETA (min)", round(y["duration_min"], 2))
#         st.metric("Predicted ETA (min)", round(preds["eta_pred"], 2))
#         st.metric("ETA MAE", round(model.eta_mae.get(), 2))

# # ---- Tab 2: Drift & Metrics ----
# with tab2:
#     st.title("📈 Model Drift & Rolling Error")

#     st.write("Real-time rolling metrics help you monitor model health over time.")

#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Fare RMSE", round(model.fare_rmse.get(), 2))
#     with col2:
#         st.metric("ETA RMSE", round(model.eta_rmse.get(), 2))

#     # 🚧 Drift Buffer Debug Info
#     st.write("ℹ️ Fare Drift Buffer Length:", len(model.fare_drift_scores))
#     st.write("ℹ️ Fare Drift Sample:", model.fare_drift_scores[:5])
#     st.write("ℹ️ ETA Drift Buffer Length:", len(model.eta_drift_scores))
#     st.write("ℹ️ ETA Drift Sample:", model.eta_drift_scores[:5])

#     # Drift Score Plot
#     st.subheader("Fare Drift Score (last 100)")
#     fig1, ax1 = plt.subplots()
#     ax1.plot(model.fare_drift_scores, label="Fare Drift", color="orange")
#     ax1.set_ylabel("Drift Score")
#     ax1.set_xlabel("Ride #")
#     ax1.grid(True)
#     st.pyplot(fig1)

#     st.subheader("ETA Drift Score (last 100)")
#     fig2, ax2 = plt.subplots()
#     ax2.plot(model.eta_drift_scores, label="ETA Drift", color="blue")
#     ax2.set_ylabel("Drift Score")
#     ax2.set_xlabel("Ride #")
#     ax2.grid(True)
#     st.pyplot(fig2)

# # ---- Tab 3: Latency Monitor ----
# with tab3:
#     st.title("⚡ Prediction Latency Insights")

#     stats = latency.summary()
#     st.metric("Min Latency (ms)", stats['min'])
#     st.metric("Max Latency (ms)", stats['max'])
#     st.metric("Avg Latency (ms)", stats['avg'])

#     st.info("This measures the time taken for a single prediction from model input to output.")

# # ---- Tab 4: Settings ----
# with tab4:
#     st.title("🛠️ Settings & Controls")

#     st.caption("Adjust simulation interval, enable/disable online updates, change models.")
#     st.write("🚧 Settings will be interactive in a later version.")


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from model.river_models import RideModel
# from utils.latency_tracker import LatencyTracker

# st.set_page_config(page_title="RideCastAI 2.0", layout="wide")

# # Load sample data
# df = pd.read_csv("data/rides.csv")
# feature_cols = ['pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon', 'distance_km', 'traffic_level']
# target_cols = ['fare_amount', 'duration_min']

# # Init model only once using session_state
# if "model" not in st.session_state:
#     st.session_state.model = RideModel()
#     st.session_state.latency = LatencyTracker()

#     # Warm-up model only once
#     for i, row in df.iloc[:20].iterrows():
#         x_warm = row[feature_cols].to_dict()
#         y_warm = row[target_cols].to_dict()
#         st.session_state.model.update(x_warm, y_warm)

# model = st.session_state.model
# latency = st.session_state.latency

# # ---- UI Tabs ----
# tab1, tab2, tab3, tab4 = st.tabs([
#     "🔮 Live Prediction", 
#     "📈 Model Drift & Metrics", 
#     "⚡ Latency Monitor", 
#     "🛠️ Settings"
# ])

# # ---- Tab 1: Live Prediction ----
# with tab1:
#     st.title("🚕 Live Fare & ETA Prediction")
#     idx = st.slider("Simulated Ride Index", 0, len(df) - 1, 0)

#     row = df.iloc[idx]
#     x = row[feature_cols].to_dict()
#     y = row[target_cols].to_dict()

#     with st.spinner("Running prediction..."):
#         preds, elapsed = latency.track(model.predict, x)
#         model.update(x, y)

#     if idx < 20:
#         st.warning("⚠️ Model is still warming up — predictions may be less accurate for early samples.")

#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Actual Fare (₹)", round(y["fare_amount"], 2))
#         st.metric("Predicted Fare (₹)", round(preds["fare_pred"], 2))
#         st.metric("Fare MAE", round(model.fare_mae.get(), 2))
#     with col2:
#         st.metric("Actual ETA (min)", round(y["duration_min"], 2))
#         st.metric("Predicted ETA (min)", round(preds["eta_pred"], 2))
#         st.metric("ETA MAE", round(model.eta_mae.get(), 2))

# # ---- Tab 2: Drift & Metrics ----
# with tab2:
#     st.title("📈 Model Drift & Rolling Error")

#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Fare RMSE", round(model.fare_rmse.get(), 2))
#     with col2:
#         st.metric("ETA RMSE", round(model.eta_rmse.get(), 2))

#     st.subheader("📉 Drift Score History")
#     fig, ax = plt.subplots(1, 2, figsize=(12, 4))

#     # Fare Drift Plot
#     ax[0].plot(model.fare_drift_scores, color='orange', label='Fare Drift')
#     ax[0].set_title("Fare Drift Score")
#     ax[0].set_ylabel("Anomaly Score")
#     ax[0].set_xlabel("Ride #")
#     ax[0].legend()
#     ax[0].grid(True)

#     # ETA Drift Plot
#     ax[1].plot(model.eta_drift_scores, color='blue', label='ETA Drift')
#     ax[1].set_title("ETA Drift Score")
#     ax[1].set_ylabel("Anomaly Score")
#     ax[1].set_xlabel("Ride #")
#     ax[1].legend()
#     ax[1].grid(True)
#     st.pyplot(fig)

#     st.subheader("📊 Rolling RMSE vs Baseline")

#     fig2, ax2 = plt.subplots(1, 2, figsize=(14, 4))
#     ax2[0].plot(model.fare_rmse_history[-100:], label="Model RMSE", color='green')
#     ax2[0].plot(model.fare_baseline_rmse[-100:], label="Baseline RMSE", color='gray', linestyle="--")
#     ax2[0].set_title("Fare RMSE Comparison")
#     ax2[0].legend()

#     ax2[1].plot(model.eta_rmse_history[-100:], label="Model RMSE", color='purple')
#     ax2[1].plot(model.eta_baseline_rmse[-100:], label="Baseline RMSE", color='gray', linestyle="--")
#     ax2[1].set_title("ETA RMSE Comparison")
#     ax2[1].legend()
#     st.pyplot(fig2)

#     st.info(f"ℹ️ Fare Drift Buffer Length: {len(model.fare_drift_scores)}")
#     st.code(model.fare_drift_scores[-5:])
#     st.info(f"ℹ️ ETA Drift Buffer Length: {len(model.eta_drift_scores)}")
#     st.code(model.eta_drift_scores[-5:])


# # ---- Tab 3: Latency Monitor ----
# with tab3:
#     st.title("⚡ Prediction Latency Insights")

#     stats = latency.summary()
#     st.metric("Min Latency (ms)", stats['min'])
#     st.metric("Max Latency (ms)", stats['max'])
#     st.metric("Avg Latency (ms)", stats['avg'])

#     st.info("This measures the time taken for a single prediction from model input to output.")

# # ---- Tab 4: Settings ----
# with tab4:
#     st.title("🛠️ Settings & Controls")

#     st.caption("Adjust simulation interval, enable/disable online updates, change models.")
#     st.write("🚧 Settings will be interactive in a later version.")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

# Init model only once using session_state
if "model" not in st.session_state:
    st.session_state.model = RideModel()
    st.session_state.latency = LatencyTracker()

    # # Warm-up model only once
    # for i, row in df.iloc[:20].iterrows():
    #     x_warm = row[feature_cols].to_dict()
    #     y_warm = row[target_cols].to_dict()
    #     st.session_state.model.update(x_warm, y_warm)

model = st.session_state.model
latency = st.session_state.latency

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

    st.info(f"ℹ️ Fare Drift Buffer Length: {len(model.fare_drift_scores)}")
    st.code(model.fare_drift_scores[-5:])
    st.info(f"ℹ️ ETA Drift Buffer Length: {len(model.eta_drift_scores)}")
    st.code(model.eta_drift_scores[-5:])

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
