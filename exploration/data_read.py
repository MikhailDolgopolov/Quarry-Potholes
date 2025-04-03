import glob
import os
import pprint
import re
from typing import Callable, Optional, List

import pandas as pd
import numpy as np
from tqdm import tqdm

from helpers import calculate_summed_magnitude, convert_dash_to_nan

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)


def read_raw_dirdata(dir_path: str, csv_pattern: str, func: Callable[[str], pd.DataFrame]) -> List[pd.DataFrame]:
    csv_pattern+='.csv'
    pattern = re.compile(csv_pattern)
    try:
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if pattern.match(f)]
        return [func(file) for file in files]
    except:
        return []

def load_preprocessed_file(file_path: str, keep_latlon=False) -> pd.DataFrame:
    """
    Reads a single preprocessed CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    keep_latlon : bool, optional
        Whether to keep 'lat' and 'lon' columns. Defaults to False.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with optional columns removed.
    """
    df = pd.read_csv(file_path, sep=';', dtype=np.float32)
    if not keep_latlon:
        df = df.drop(columns=['lat', 'lon'], errors='ignore')
    return df

def load_preprocessed(folder_path:str, keep_latlon=False, sample_frac=1)->pd.DataFrame:
    """
        Loads all preprocessed CSV files from a folder and optionally downsamples them.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing CSV files.
        keep_latlon : bool, optional
            Whether to keep 'lat' and 'lon' columns. Defaults to False.
        sample_frac : float, optional
            Fraction of data to sample. Defaults to 1 (no downsampling).

        Returns
        -------
        pd.DataFrame
            Combined and optionally sampled DataFrame from all CSVs.
        """
    dataframes = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            dataframes.append(load_preprocessed_file(file_path, keep_latlon))

    result = pd.concat(dataframes, ignore_index=True).sample(frac=sample_frac)
    return result

def read_track(path: str) -> Optional[pd.DataFrame]:
    try:
        raw_df = pd.read_csv(path, delimiter=';', encoding='windows-1251', index_col=0)
    except Exception as e:
        print(f"Trouble with {path}:")
        print(e)
        return None
    explicit_columns = ['Широта', 'Долгота', 'Скорость', 'point']
    pattern_columns = raw_df.columns[raw_df.columns.str.contains('Ускорение|наклон', regex=True)]

    # Combine columns
    selected_columns = explicit_columns + list(pattern_columns)

    new_names = ['lat', 'lon', 'vel', 'class', 'acc_X', 'acc_Y', 'acc_Z', 'fb_tilt',
                 'tilt']
    # print(dict(zip(selected_columns, new_names)))
    names_map = {selected_columns[i]: new_names[i] for i in range(len(selected_columns))}
    try:
        # Filter and rename DataFrame
        filtered_df = raw_df[selected_columns]
        df = filtered_df.rename(columns=names_map)
    except Exception as e:
        print(f"Trouble with {path}:")
        print(e)
        return None

    recentered = ['acc_X', 'acc_Y', 'acc_Z', 'fb_tilt', 'tilt']

    for fix in recentered:
        mean = df[fix].mean()
        if abs(round(mean))>0:
            df[fix] = df[fix] - round(mean)
            # print(fix)

    df['class'] = df['class'].str.replace(r'.*?(\d+).*', r'\1', regex=True)
    df['acc'] = calculate_summed_magnitude(df, 'acc_')

    df = convert_dash_to_nan(df)
    df['class'] = df['class'].fillna(0)
    df['hole'] = np.where(df['class']>0, 1, 0)
    return df