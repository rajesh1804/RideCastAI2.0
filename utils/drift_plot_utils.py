import matplotlib.pyplot as plt

def plot_drift_with_overlay(ax, scores, drift_flags, label="Drift", color="orange"):
    ax.step(range(len(scores)), scores, where="mid", label=f"{label} Drift Score", color=color)
    
    for i, flag in enumerate(drift_flags):
        if flag:
            ax.axvline(i, color='red', linestyle='--', alpha=0.6)
    
    ax.set_title(f"{label} Input Drift w/ Output Flags")
    ax.set_ylabel("Score")
    ax.set_xlabel("Sample Index")
    ax.legend()
