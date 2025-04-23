import glob
import os
import pprint
import re
from pathlib import Path
from typing import Callable, Optional, List, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

from helpers import calculate_summed_magnitude, convert_dash_to_nan

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

straight_predictors = ['acc_X', 'acc_Y', 'acc_Z', 'acc','fb_tilt', 'tilt']
def load_engineered_data(dir_path: Path|str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(dir_path, 'route*.csv'))
    dfs = []
    for csv_file in tqdm(csv_files, desc=f'Loading from {dir_path}'):
        df = pd.read_csv(csv_file)
        dfs.append(df)
    points = pd.concat(dfs, ignore_index=True)
    return points


def read_dir_csvs(dir_path: Path|str, func: Callable[[str], pd.DataFrame], csv_pattern: str=r'.*_w') -> List[pd.DataFrame]:
    csv_pattern+=r'\.csv'
    pattern = re.compile(csv_pattern)
    try:
        # print(os.listdir(dir_path))
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if pattern.match(f)]
        return [func(file) for file in files]
    except:
        return []

def read_recoded_track(path: str, delimiter: str = ',') -> Optional[pd.DataFrame]:
    try:
        raw_df = pd.read_csv(path, sep=delimiter, encoding='utf-8')
    except Exception as e:
        print(f"Trouble with {path}:")
        print(e)
        return None
    explicit_columns = ['Широта', 'Долгота', 'Скорость', 'pothole', 'severity']
    pattern_columns = raw_df.columns[raw_df.columns.str.contains('Ускорение|наклон', regex=True)]

    # Combine columns
    selected_columns = explicit_columns + list(pattern_columns)

    new_names = ['lat', 'lon', 'vel', 'pothole', 'severity', 'acc_X', 'acc_Y', 'acc_Z', 'fb_tilt',
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
        if abs(round(mean)) > 0:
            df[fix] = df[fix] - round(mean)

    return df


def read_raw_track(path: str|Path, delimiter: str = ';', encoding='windows-1251') -> Optional[pd.DataFrame]:
    try:
        raw_df = pd.read_csv(path, sep=delimiter, encoding=encoding)
    except Exception as e:
        print(f"Trouble with {path}:")
        print(e)
        return None
    explicit_columns = ['Широта', 'Долгота', 'Скорость',
                        'Давление левый передний цилиндр', 'Давление правый передний цилиндр',
                        'Давление левый задний цилиндр', 'Давление правый задний цилиндр',
                        'point']
    pattern_columns = raw_df.columns[raw_df.columns.str.contains('Ускорение|наклон', regex=True)]

    # Combine columns
    selected_columns = explicit_columns + list(pattern_columns)

    new_names = ['lat', 'lon', 'vel',
                 'pressure_FL', 'pressure_FR', 'pressure_RL', 'pressure_RR',
                 'severity', 'acc_X', 'acc_Y', 'acc_Z', 'fb_tilt', 'tilt']
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

    df['severity'] = df['severity'].str.replace(r'.*?(\d+).*', r'\1', regex=True)
    df['acc'] = calculate_summed_magnitude(df, 'acc_')

    df = convert_dash_to_nan(df)
    df['severity'] = df['severity'].fillna(0)
    df['pothole'] = np.where(df['severity']>0, 1, 0)
    return df

def load_plain_data(folder: str | Path, routes: range = range(1, 39)):
    routes_dfs = []
    for i in routes:
        # Load all CSVs from the directory into a list of DataFrames
        dfs = read_dir_csvs(Path(folder) / f'route{i}', pd.read_csv)

        if len(dfs) > 0:
            route = pd.concat(dfs, ignore_index=True)
            routes_dfs.append(route)

    # Concatenate all route DataFrames into one final DataFrame
    return pd.concat(routes_dfs, ignore_index=True)

def reread_raw_data(routes: range,
                paths_func: Callable[[int], str],
                output_folder: str):

    routes_dirs = [paths_func(i) for i in routes]
    for i, route in enumerate(tqdm(routes_dirs)):
        tracks = read_dir_csvs(route, read_raw_track, r'[0-9]{1,3}_w')
        tracks = [track  for track in tracks if track is not None and not track.empty]
        os.makedirs(Path(output_folder) / f"route{i+1}", exist_ok=True)
        for j, track in enumerate(tracks):
            track.to_csv(Path(output_folder) / f"route{i+1}/"/f"{j+1}_w.csv", index=False)

if __name__ == '__main__':
    # reread_raw_data(range(1, 39),
    #             lambda x: f"data/input-raw/route{x}",
    #             f"data/renamed")

    dfs = load_plain_data(f"data/renamed")

    print(dfs['pothole'].value_counts())
