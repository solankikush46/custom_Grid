# plot_metrics.py

import os
from constants import *
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv(csv_path, output_dir, rolling_window=2000):
    """
    Reads a CSV file, applies a rolling mean, and generates line plots for numeric columns.

    Args:
        csv_path (str): Path to the CSV file.
        output_dir (str): Directory to save the PNG plots.
        rolling_window (int): Window size for rolling mean smoothing.
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Skipping empty CSV file: {csv_path}")
            return []
    except pd.errors.EmptyDataError:
        print(f"Skipping empty or malformed CSV file: {csv_path}")
        return []

    print(f"\nLoaded {csv_path}: {df.shape[0]} rows, columns: {list(df.columns)}")

    # Create the plots directory
    plot_dir = output_dir

    # Extract base name without extension
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    numeric_cols = df.select_dtypes(include='number').columns

    plots = []

    for col in numeric_cols:
        smoothed = df[col].rolling(window=rolling_window, min_periods=1).mean()

        plt.figure(figsize=(10, 4))
        plt.plot(smoothed)
        plt.title(f"{base_name}: {col} (Rolling Mean window={rolling_window})")
        plt.xlabel("Row")
        plt.ylabel(col)
        plt.grid(True)

        # Clean column name
        safe_col_name = col.replace("/", "_").replace(" ", "_")

        # Build output file name
        out_path = os.path.join(plot_dir, f"{base_name}_{safe_col_name}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        plots.append(out_path)

        print(f"Saved plot: {out_path}")

    return plots

def plot_all_metrics(
    log_dir,
    output_dir,
    rolling_window=2000
):
    """
    Generates plots for all metrics CSVs in a log directory.

    Args:
        log_dir (str): Directory containing CSV files.
        output_dir (str): Directory to save PNG plots.
        rolling_window (int): Window size for rolling mean smoothing.

    Returns:
        dict: Mapping of CSV filenames to lists of saved plot file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_plots = {}

    # Look for all metrics CSVs
    csv_files = [
        f for f in os.listdir(log_dir)
        if f.endswith(".csv") and (
            f.startswith("timestep_metrics") or
            f.startswith("episode_metrics") or
            f.startswith("subrewards_metrics")
        )
    ]

    for f in csv_files:
        csv_path = os.path.join(log_dir, f)
        plots = plot_csv(csv_path, output_dir, rolling_window=rolling_window)
        all_plots[f] = plots

    return all_plots

def generate_all_plots(base_dir=None):
    """
    Recursively search for folders containing metrics CSVs
    and generate plots for each.
    """
    if base_dir is None:
        base_dir = LOG_DIR
    for root, dirs, files in os.walk(base_dir):
        csvs = [f for f in files if f.endswith(".csv") and (
            f.startswith("timestep_metrics") or
            f.startswith("episode_metrics") or
            f.startswith("subrewards_metrics")
        )]

        if csvs:
            print(f"\n=== Processing CSVs in {root} ===")
            output_dir = os.path.join(root, "plots")
            plots = plot_all_metrics(log_dir=root, output_dir=output_dir)
            print(f"Generated {sum(len(v) for v in plots.values())} plots in {output_dir}")
