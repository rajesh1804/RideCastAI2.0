---
title: "RideCastAI2.0"
emoji: "🚕"
colorFrom: "red"
colorTo: "yellow"
sdk: streamlit
sdk_version: "1.35.0"
app_file: ui/app.py
pinned: true
---

# 🚕 RideCastAI 2.0 — Real-Time Fare & ETA Predictor with Drift Recovery and Latency Optimization

[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red?logo=streamlit)](https://streamlit.io)  
[![Real-Time ML](https://img.shields.io/badge/ML-Online%20Learning-blue?logo=scikit-learn)]()  
[![Deployed on Hugging Face](https://img.shields.io/badge/Hosted%20on-HuggingFace-orange?logo=huggingface)](https://huggingface.co/spaces/rajesh1804/RideCastAI2.0)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🚦 **RideCastAI 2.0** is a production-grade real-time ML system that predicts ride fares & ETAs, detects input/output drift, updates models live, and tracks latency — simulating a dynamic dispatch engine like Uber, Lyft, or Grab.

🎯 Designed for top-tier AI infrastructure roles where **adaptability, performance, and monitoring** are key.

---

## ⚡️ What’s New in 2.0

| Feature                     | 1.0                               | 🚀 2.0 Upgrade                                    |
|----------------------------|-----------------------------------|--------------------------------------------------|
| Prediction Scope           | Static fare & ETA prediction      | 🔁 Real-time prediction with async ingestion     |
| Drift Detection            | None                              | ✅ Input + Output drift with visual alerts       |
| Model Adaptation           | Offline model                     | ✅ Online learning with `river`                  |
| Caching & Inference        | None                              | ✅ ONNX + Joblib caching + asyncio               |
| Latency Tracking           | No                                | ✅ Real-time latency chart                       |
| Error Debugging            | No                                | ✅ Live top-5 worst predictions                  |
| UI                         | Basic chart                       | ✅ Tabbed Streamlit dashboard with toggles       |
| Deployability              | Basic                             | ✅ Free-tier deployable on Hugging Face          |

---

## 📊 Architecture Overview

<p align="center">
  <img src="https://github.com/rajesh1804/RideCastAI2.0/raw/main/assets/RideCastAI2.0-architecture.png" alt="RideCastAI2.0 System Flow" width="750"/>
</p>

```text
Rides (CSV/Simulated) 
→ Feature Preprocessing
→ Model Inference (ONNX / River)
→ Caching Layer (Joblib)
→ Output Metrics Tracker (MAE / RMSE / Drift)

↳ Input Drift Detector (HalfSpaceTrees)
↳ Output Drift Detectors (KSWIN, ADWIN)
↳ Online Learner (River LinearRegression)
↳ Latency Tracker (Async + Retry logic)

↳ Streamlit UI with Live Dashboard:
   - Prediction + Drift + Latency Tabs
   - Drift Injection / Online Update Toggle
   - Top-5 Error Viewer
   - Logs + Trace Overlay
```

---

## 🌟 Key Features

✅ **Real-Time Fare & ETA Prediction**  
✅ **Online Learning (River)** — self-adapts to incoming data  
✅ **Input Drift Detection** (HalfSpaceTrees)  
✅ **Output Drift Detection** (KSWIN, ADWIN)  
✅ **ONNX-Optimized Model + Joblib Caching**  
✅ **Live Top-5 Worst Prediction Viewer**  
✅ **Latency Visualization per Batch**  
✅ **Drift Injection Toggle for Testing**  
✅ **Fully Modular Streamlit UI**  
✅ **Free-tier Deployable** (No LLMs required)

---

## 🧠 Component Stack

> Each module is modular, production-grade, and latency-aware.

| Component            | Role                                         | Tech Stack                |
|----------------------|----------------------------------------------|---------------------------|
| 🚕 Predictor          | Predicts fare & ETA                         | `scikit-learn`, `onnx`, `river` |
| 🧠 Drift Detectors    | Input (HalfSpaceTrees), Output (KSWIN/ADWIN) | `river.drift`             |
| ♻️ Online Learner     | Updates model weights per ride              | `river.linear_model.LinearRegression` |
| 💾 Caching            | Stores past predictions for reuse           | `joblib`                  |
| 🕒 Latency Tracker    | Logs inference time and averages            | `time`, `asyncio`         |
| 📈 Visual Overlay     | RMSE, Drift Flags, Top-5 Errors             | `matplotlib`, `seaborn`   |
| 🧪 Drift Injector     | Force anomaly to test system recovery       | Manual + Config toggle    |
| 🖥️ UI Layer           | Live dashboard with tabs                    | `Streamlit`               |

---

## 🔎 How It Works

- Ingests a real-time ride (or simulated stream)
- Featurizes and routes through predictor
- Monitors:
  - 🚨 Input Drift via HalfSpaceTrees
  - 📉 Output Drift via KSWIN & ADWIN
- Updates model weights online (if enabled)
- Caches previous rides to reduce inference cost
- Logs latency, RMSE, drift spikes, and error outliers
- Visualizes all data on an elegant Streamlit UI

---

## 🖼️ UI Preview

<p align="center">
  <img src="https://github.com/rajesh1804/RideCastAI2.0/raw/main/assets/RideCastAI2.0-demo.gif" width="750"/>
</p>

**Tabbed layout includes:**

- 🔮 **Live Prediction** — Real-time results + top-5 worst errors  
- ⚠️ **Drift & Metrics** — Input + Output drift tracking + RMSE overlay  
- ⚡ **Latency Monitor** — Inference timing graph  
- 🔧 **Settings** — Inject Drift, Enable Online Learning, View Architecture

📌 Try the live app here:  
👉 [RideCastAI 2.0 – Hugging Face Space](https://huggingface.co/spaces/rajesh1804/RideCastAI2.0)

---

## 🧪 Sample Output (Ride ID: `ride_1027`)

- **🚕 Predicted Fare**: ₹184.76  
- **🚕 Predicted ETA**: 12.4 minutes  
- **📊 RMSE (Last 50)**: Fare: ₹9.12 | ETA: 1.8 mins  
- **📉 Drift**:  
  - Input: ❌ No  
  - Output: ✅ ADWIN Triggered  
- **Latency**: 1.2s  
- **In Top-5 Error**: ✅ Yes → Underestimated ETA by 4.9 mins

---

## 📁 Project Structure

```bash
RideCastAI2.0/
├── ui/
│   └── app.py                     # Streamlit frontend
├── model/
│   └── river_models.py            # Drift + model logic
├── utils/
│   ├── latency_tracker.py
│   └── drift_plot_utils.py
├── data/
│   └── rides.csv                  # Input data
├── assets/
│   ├── ridecast_architecture.png  # Arch Diagram
│   └── ridecast_demo.gif          # UI Demo
└── requirements.txt
```

---

## 💼 Why This Project Stands Out

- ✅ Real-time architecture, not just static ML
- ✅ Combines **drift detection + online learning + latency awareness**
- ✅ Debuggable like internal Uber/Lyft tools
- ✅ Clean, modular UI with tabbed monitoring
- ✅ Designed for **free-tier deployability** with zero cost
- ✅ Built as a candidate portfolio app to demonstrate elite ML engineering

---

## 🧰 Run Locally (for Devs)

```bash
git clone https://github.com/rajesh1804/RideCastAI2.0
cd RideCastAI2.0
pip install -r requirements.txt
streamlit run ui/app.py
```

---

## 🧠 Linked Projects

| Project              | Description                                                              | Link |
|----------------------|---------------------------------------------------------------------------|------|
| 🧵 ThreadNavigatorAI 2.0 | Multi-Agent Reddit thread analyzer with LLM-as-a-Judge                  | [🔗 View](https://github.com/rajesh1804/ThreadNavigatorAI2.0) |
| 🛒 GroceryGPT+          | Vector search + reranking grocery assistant with fuzzy recall             | [🔗 View](https://github.com/rajesh1804/GroceryGPT) |
| 🎬 StreamWiseAI         | Netflix-style movie recommender with Retention Coach Agent                | [🔗 View](https://github.com/rajesh1804/StreamWiseAI) |

---

## ⚠️ Known Challenges

- 🧱 `river` failed to install on Hugging Face (Rust required) → solved via downgrade to `river==0.15.1`
- 🌀 Drift injection needed careful UI/UX isolation — solved with toggle + sidebar logs
- 🧠 ONNX conversion limited to scikit-learn baseline only (not River) — handled fallback with hybrid logic
- 🕒 Async scheduler in Streamlit was tricky — solved using `asyncio.create_task` + stateful cache

---

## 🧑‍💼 About Me

**Rajesh Marudhachalam** — AI/ML Engineer building real-time, agentic, and production-grade AI systems.  
📍 [GitHub](https://github.com/rajesh1804) | [LinkedIn](https://www.linkedin.com/in/rajesh1804/)

Projects: [ThreadNavigatorAI2.0](https://github.com/rajesh1804/ThreadNavigatorAI2.0), [StreamWiseAI](https://github.com/rajesh1804/StreamWiseAI), [GroceryGPT+](https://github.com/rajesh1804/GroceryGPT)

---

## 🙌 Acknowledgments

- [River](https://riverml.xyz) — Streaming ML + Drift Detection  
- [Streamlit](https://streamlit.io) — UI Framework  
- [Hugging Face Spaces](https://huggingface.co/spaces) — App Hosting

---

## 📜 License

MIT License

⭐️ *Star this repo if it impressed you. Follow for more production-grade ML builds.*
