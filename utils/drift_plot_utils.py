# import pandas as pd
# import altair as alt

# def get_drift_chart(fare_drift, eta_drift, window=20):
#     df = pd.DataFrame({
#         'ride_idx': list(range(len(fare_drift))),
#         'Fare Drift': fare_drift,
#         'ETA Drift': eta_drift
#     })

#     df['Fare Rolling'] = df['Fare Drift'].rolling(window=window).mean()
#     df['ETA Rolling'] = df['ETA Drift'].rolling(window=window).mean()

#     fare_chart = alt.Chart(df).mark_line(color='orange').encode(
#         x='ride_idx', y='Fare Drift'
#     ).properties(title="Fare Drift Score")

#     fare_roll = alt.Chart(df).mark_line(color='red', strokeDash=[4,2]).encode(
#         x='ride_idx', y='Fare Rolling'
#     )

#     eta_chart = alt.Chart(df).mark_line(color='blue').encode(
#         x='ride_idx', y='ETA Drift'
#     ).properties(title="ETA Drift Score")

#     eta_roll = alt.Chart(df).mark_line(color='purple', strokeDash=[4,2]).encode(
#         x='ride_idx', y='ETA Rolling'
#     )

#     final_chart = alt.hconcat(fare_chart + fare_roll, eta_chart + eta_roll).resolve_scale(y='independent')
#     return final_chart

import matplotlib.pyplot as plt

def plot_drift_with_overlay(ax, drift_scores, drift_flags=None, label="Drift", color="orange", rolling_window=10):
    """
    Plot input drift scores with rolling average and output drift flags on a given matplotlib Axes.

    Args:
        ax (matplotlib.axes.Axes): The axes object to draw the plot on.
        drift_scores (list of float): Input drift scores (e.g., from HalfSpaceTrees).
        drift_flags (list of bool): Optional binary flags indicating output drift.
        label (str): Label for the input drift line.
        color (str): Color for the input drift line.
        rolling_window (int): Rolling average window size for smoother trends.
    """
    ax.plot(drift_scores, label=f"{label} Input Drift", color=color, linewidth=1.5)

    if len(drift_scores) >= rolling_window:
        rolling_avg = [
            sum(drift_scores[max(0, i - rolling_window):i + 1]) / min(i + 1, rolling_window)
            for i in range(len(drift_scores))
        ]
        ax.plot(rolling_avg, color="black", linestyle="--", linewidth=1, label="Rolling Avg")

    if drift_flags:
        for i, is_drift in enumerate(drift_flags):
            if is_drift:
                ax.axvline(x=i, color="red", linestyle=":", linewidth=0.8)

    ax.set_xlabel("Ride Index")
    ax.set_ylabel("Drift Score")
    ax.set_title(f"{label} Drift (Input + Output Flags)")
    ax.legend()
    ax.grid(True)
