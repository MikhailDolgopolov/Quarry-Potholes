import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from exploration.data_read import read_raw_dirdata, read_track, load_preprocessed_file

def lerp_dataframe(df: pd.DataFrame, lerp_steps: int) -> pd.DataFrame:
    """
    Linearly interpolates a DataFrame to add `lerp_steps` intermediate points between each row.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time-series data.
    lerp_steps : int
        Number of interpolation steps per interval.
        Example: lerp_steps=5 means adding 4 points between two original rows (5 Hz instead of 1 Hz).

    Returns
    -------
    pd.DataFrame
        Interpolated DataFrame with higher temporal resolution.
    """
    if lerp_steps <= 1 or len(df) < 2:
        return df  # No interpolation needed.

    original_length = len(df)
    new_length = (original_length - 1) * lerp_steps + 1
    new_index = np.linspace(0, original_length - 1, new_length)

    df = df.copy()
    df.index = np.arange(original_length)  # Ensure the index is numeric
    df_interp = df.reindex(new_index).interpolate(method='linear')
    df_interp.reset_index(drop=True, inplace=True)

    return df_interp

def lerp_data(
        routes,
        output_folder,
        paths_func: Callable[[int], str],
        dir_pattern=r'[0-9]{1,3}_w',
        lerp_steps=5
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for route in tqdm(routes, desc="Processing routes"):
        dir_name = paths_func(route)
        tracks_data = read_raw_dirdata(dir_name, dir_pattern, lambda x: read_track(x))
        # print(dir_name)
        if not tracks_data:  # Skip if no data
            continue
        lerp_route_path = output_folder / f'route{route}'
        os.makedirs(lerp_route_path, exist_ok=True)
        for i, df in enumerate(tracks_data):
            if df.empty:
                continue
            df_final = lerp_dataframe(df, lerp_steps)
            df_final.to_csv(lerp_route_path / f'{i+1}_w.csv', index=False, sep=';')


if __name__ == "__main__":
    routes = range(1, 36)
    dir_path_func = lambda n: f"data/routes/route{n}"
    output_folder = f"data/routes-lerp"
    os.makedirs(output_folder, exist_ok=True)

    # Run preprocessing with interpolation; adjust lerp_steps as needed.
    lerp_data(
        routes=routes,
        output_folder=output_folder,
        paths_func=dir_path_func,
        lerp_steps=5
    )

    # df = load_preprocessed_file('data/lerp/route1.csv')
    # df = load_preprocessed_file('data/hole0/route1.csv')
    #
    # # print(df.describe())
    #
    # plt.plot(df['hole'])
    # plt.show()
