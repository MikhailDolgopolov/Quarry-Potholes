import os
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from exploration.data_read import read_dir_csvs

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)


def assign_threshold(input_df: pd.DataFrame, measure_col: str, base_level: float,
                     lower_mult: float, upper_mult: float, marker_mult: float, marker_col: str) -> pd.DataFrame:
    """
    For rows in input_df where the value in measure_col is between base_level * lower_mult
    and base_level * upper_mult, assign in marker_col the value base_level * marker_mult.
    """
    condition = (input_df[measure_col] > base_level * lower_mult) & (input_df[measure_col] < base_level * upper_mult)
    input_df.loc[condition, marker_col] = base_level * marker_mult
    return input_df


def adjust_marker_values(df: pd.DataFrame, marker_col: str) -> pd.DataFrame:
    """
    Adjusts the marker column based on its neighboring values.
    (This replicates the idea of shifting from the original function.)
    """
    # Create shifted versions (temporary)
    df['marker_next'] = df[marker_col].shift(-1)
    df['marker_prev'] = df[marker_col].shift(1)

    df[marker_col] = df.apply(
        lambda x: x['marker_next']
        if ((x[marker_col] < 1.3 * x['marker_next']) and (x['marker_prev'] < x[marker_col] > x['marker_next']))
           or ((pd.isna(x['marker_prev'])) and (x[marker_col] > x['marker_next']))
           or ((x['marker_prev'] > x[marker_col]) and (x[marker_col] < x['marker_next']))
        else x[marker_col],
        axis=1
    )
    df.drop(['marker_next', 'marker_prev'], axis=1, inplace=True)
    return df


def assign_marker_labels(df: pd.DataFrame, measure_col: str, marker_col: str, base_level: float,
                         thresholds: dict) -> pd.DataFrame:
    """
    Create separate columns with marker labels based on the final marker value.
    thresholds should be a dict mapping marker multiplier to a label suffix.
    """
    for mult, label_suffix in thresholds.items():
        new_col = measure_col + '_' + label_suffix
        df[new_col] = df.apply(lambda x: f'point_{label_suffix}{measure_col}'
                                 if x[marker_col] == base_level * mult else '-', axis=1)
    return df


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
        DataFrame with added 'severity' and 'pothole' columns.
    """
    aggregation = {
        'front': ['Давление левый передний цилиндр', 'Давление правый передний цилиндр'],
        'back': ['Давление левый задний цилиндр', 'Давление правый задний цилиндр']
    }

    threshold = 1.3
    severity_columns = []

    for side, columns in aggregation.items():
        for col in columns:
            # Ensure the column is numeric
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Compute rolling average of both columns and take their mean
        side_mean = input_df[columns].rolling(window=spike_window, center=True).mean().mean(axis=1)

        for col in columns:
            ratio = input_df[col] / side_mean
            severity_col = f"{col}_severity"
            input_df[severity_col] = np.where(ratio > threshold, ratio, 0)
            severity_columns.append(severity_col)

    # Aggregate
    input_df["severity"] = input_df[severity_columns].max(axis=1)
    input_df["pothole"] = (input_df["severity"] > 0).astype(int)

    # Optional: drop the intermediate columns
    input_df.drop(columns=severity_columns, inplace=True)

    return input_df



if __name__ == '__main__':
    ws = 50
    routes = range(1, 36)
    out_folder = Path(f'data/input-recoded/{ws}peaks')
    os.makedirs(out_folder, exist_ok=True)
    for r_id in tqdm(routes):
        in_folder = f'data/input-raw/route{r_id}'
        read_raw = lambda x: pd.read_csv(x, sep=';', encoding='cp1251', index_col=0)
        frames = read_dir_csvs(in_folder, read_raw, r'.*_w')
        route_folder = out_folder / f'route{r_id}'
        os.makedirs(route_folder, exist_ok=True)
        for i, df in enumerate(frames):
            if df.shape[0] > 0:
                marked_df = pit_marking(df.copy(), ws)
                marked_df.to_csv(Path(out_folder)/f'route{r_id}/{i+1}_w.csv', index=False)
