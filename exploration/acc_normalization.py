import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from exploration.data_read import load_plain_data, read_dir_csvs


def speed_conditional_zscores(df, speed_col='vel', n_bins=20, measure_cols=None, abs_z=True):
    """
    Computes speed-conditional z-scores for the given accelerometer columns.
    Each z-score is computed as:
        z = (x – mean(speed_bin)) / std(speed_bin)

    Parameters
    ----------
    df : pd.DataFrame
    speed_col : str
        Name of the speed column.
    n_bins : int
        Number of speed quantile bins.
    measure_cols : list[str] or None
        Columns to normalize. If None, all columns starting with 'acc' are used.
    abs_z : bool
        Whether to return absolute z-scores.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with z-score columns added (one per measure).
    """
    df = df.copy()

    if measure_cols is None:
        measure_cols = [col for col in df.columns if col.startswith('acc')]

    # Bin the speed column into quantiles
    df['speed_bin'] = pd.qcut(df[speed_col], q=n_bins, duplicates='drop')

    # Compute mean and std per speed_bin for each measure
    agg = df.groupby('speed_bin', observed=True)[measure_cols].agg(['mean', 'std'])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]

    # Merge stats back to original DataFrame
    df = df.join(agg, on='speed_bin')

    # Compute z-scores
    z_data = {}
    for col in measure_cols:
        mu = df[f"{col}_mean"]
        sigma = df[f"{col}_std"].replace(0, np.nan)
        z = (df[col] - mu) / sigma
        if abs_z:
            z = np.abs(z)
        z_data[f"{col}_zscore"] = z.fillna(0)

    z_df = pd.DataFrame(z_data, index=df.index)

    # Combine z-scores with original dataframe (excluding original measure columns)
    df = pd.concat([df.drop(columns=measure_cols), z_df], axis=1)
    df = df.drop(columns=[col for col in df.columns if col.endswith('_mean') or col.endswith('_std')])
    df = df.drop(columns='speed_bin')

    return df


def normalize_acc(input_dir: str | Path, speed_col: str = 'vel', n_bins: int = 20):
    input_dir = Path(input_dir)
    name = input_dir.name
    output_dir = Path(f'data/normalized/{name}_norm')
    folders = [f for f in input_dir.iterdir() if 'route' in f.name and f.is_dir()]

    for route_folder in tqdm(folders, desc=f'Normalizing {name}'):
        track_dfs = read_dir_csvs(route_folder, pd.read_csv)
        output_path = output_dir / route_folder.name
        output_path.mkdir(parents=True, exist_ok=True)

        for i, track_df in enumerate(track_dfs):
            if track_df.empty:
                continue
            normalized_df = speed_conditional_zscores(track_df, speed_col, n_bins)
            normalized_df.to_csv(output_path / f'{i + 1}_w.csv', index=False)


if __name__ == '__main__':
    for ws in [30]:
        normalize_acc(input_dir=f'data/relabeled/extremes_w{ws}')

