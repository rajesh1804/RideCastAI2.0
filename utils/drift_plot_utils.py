import matplotlib.pyplot as plt

def plot_drift_with_overlay(ax, drift_scores, drift_flags=None, label="Drift", color="orange", rolling_window=10):
    """
    Modular drift plotter: shows input drift, rolling avg, and flags from output drift detectors.
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
                ax.axvline(x=i, color="red", linestyle=":", linewidth=1)

    ax.set_xlabel("Ride Index")
    ax.set_ylabel("Drift Score")
    ax.set_title(f"{label} Drift (Input + Output)")
    ax.legend()
    ax.grid(True)
