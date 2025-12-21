import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
from datetime import datetime
from logger_utils import log_message

def plot_traces_3d_dist(traces, freqs, distance, pulse_width):
    """
    Plot 3D BGS with correct distance scaling based on pulse width.
    
    Args:
        traces: 2D array of traces (freq x distance)
        freqs: Frequency array
        distance: Fiber length in meters (for reference, not used in calculation)
        pulse_width: Pulse width in nanoseconds
    """
    n_steps, n_samples = traces.shape
    
    # Calculate distance scaling based on pulse width
    # Assuming 200 MS/s sample rate (should match your acquisition)
    sample_rate = 200e6  # Hz
    group_size = int(pulse_width * sample_rate * 1e-9)
    sampling_period = 1 / sample_rate
    v_fiber = 2e8  # m/s (speed of light in fiber)
    
    # Distance per point after spatial averaging
    distance_scale_m = (group_size * sampling_period * v_fiber) / 2
    distance_m = np.arange(n_samples) * distance_scale_m
    
    # Meshgrid for plotting
    D, F = np.meshgrid(distance_m, freqs)

    # 3D Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(D, F, traces, cmap="viridis", linewidth=0, antialiased=True)

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Frequency (GHz)")
    ax.set_zlabel("Amplitude")
    ax.set_title(f"3D Traces vs Frequency (PW={pulse_width}ns, Res={distance_scale_m:.2f}m)")

    fig.colorbar(surf, shrink=0.5, aspect=10, label="Amplitude")
    plt.show()


def save_traces(traces, freqs, t_us, save_dir, prefix="processed", log_file=None):
    # Folder named by today's date (YYYY-MM-DD)
    today_str = datetime.today().strftime("%Y-%m-%d")
    day_folder = os.path.join(save_dir, today_str)

    # Create folder if not exists
    if not os.path.exists(day_folder):
        os.makedirs(day_folder)
        print(f"Created folder: {day_folder}")
    else:
        print(f"Using existing folder: {day_folder}")

    # Timestamp for filename
    timestamp = datetime.now().strftime("%H%M%S")

    # Build DataFrame:
    # rows = freqs, columns = time values
    df = pd.DataFrame(traces, index=freqs, columns=np.round(t_us, 5))
    df.index.name = "Frequency"

    # Filepath
    filename = f"{prefix}_{timestamp}.csv"
    filepath = os.path.join(day_folder, filename)

    # Save to CSV
    df.to_csv(filepath)

    print(f"Saved matrix CSV: {filepath}")
    
    # Log the file saving event if log_file is provided
    if log_file:
        log_message(f"Saved trace data to: {filepath}", log_file=log_file)


def save_raw_traces(raw_traces, freqs, cfg, save_dir, tag="raw"):
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{tag}_traces_{timestamp}.npz"
    fpath = os.path.join(save_dir, fname)

    np.savez_compressed(
        fpath,
        raw_traces=raw_traces,
        freqs=freqs,
        cfg=cfg
    )

    print(f"✅ Raw traces saved to:\n{fpath}")
    return fpath