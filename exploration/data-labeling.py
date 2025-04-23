import glob
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from exploration.data_read import read_dir_csvs

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)


def pit_marking(input_df: pd.DataFrame, spike_window: int) -> pd.DataFrame:
    """
    Detects pressure spikes as potential potholes based on ratio of measured value to rolling average.

    Any value above 1.3x the rolling average is considered a pothole, and severity is the ratio itself.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame containing raw pressure measurements.
    spike_window : int
        Window size for rolling mean.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'severity' column.
    """
    aggregation = {
        'front': ['pressure_FL', 'pressure_FR'],
        'back': ['pressure_RL', 'pressure_RR'],
    }
    severity_columns = []
    pressure_columns = np.array([*aggregation.values()]).flatten()
    for part, columns in aggregation.items():
        for col in columns:
            # Ensure the column is numeric
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Compute rolling average of both columns and take their mean
        side_mean = input_df[columns].rolling(window=spike_window, center=True).mean().mean(axis=1)

        for col in columns:
            severity_col = f"{col}_severity"
            input_df[severity_col] = np.abs(1-input_df[col] / side_mean)
            severity_columns.append(severity_col)

    # Aggregate
    input_df["severity"] = input_df[severity_columns].max(axis=1).fillna(0)
    input_df.drop(columns='pothole', inplace=True, errors='ignore')

    input_df.drop(columns=severity_columns, inplace=True)
    input_df.drop(columns=pressure_columns, inplace=True)

    return input_df

def relabel_data(extremes_window_size: int, input_folder: Path|str,
                 output_folder: Path|str, routes: range = range(1, 39)) -> None:
    """
    Takes all the CSV files from input_folder, re-labels it based on pressure spikes,
     and saves new CSVs to output_folder.
    """
    output_folder = Path(output_folder)/f'extremes_w{extremes_window_size}'
    os.makedirs(output_folder, exist_ok=True)
    for r_id in tqdm(routes, desc=f'Relabeling across window size {extremes_window_size}'):
        in_folder = Path(input_folder)/f'route{r_id}'
        frames = read_dir_csvs(in_folder, pd.read_csv, r'.*_w')
        route_folder = Path(output_folder) / f'route{r_id}'
        os.makedirs(route_folder, exist_ok=True)
        for i, df in enumerate(frames):
            if df.shape[0] > 0:
                marked_df = pit_marking(df.copy(), extremes_window_size)
                marked_df.to_csv(route_folder / f'{i + 1}_w.csv', index=False)


if __name__ == '__main__':
    # WSs = [10, 20, 30, 40, 50]
    WSs = [30]
    for ws in WSs:
        relabel_data(ws, 'data/renamed', 'data/relabeled')


