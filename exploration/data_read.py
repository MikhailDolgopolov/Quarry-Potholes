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


def read_truck_data(path: str) -> Optional[pd.DataFrame]:
    try:
        #
        raw_df = pd.read_csv(path, delimiter=';', encoding='windows-1251', index_col=0)
        raw_df = convert_dash_to_nan(raw_df)
    except:
        return None
    explicit_columns = ['Широта', 'Долгота', 'Скорость']
    hole_col=''
    for c in ['nom_point', 'nom_hole']:
        if c in raw_df:
            hole_col=c
    pattern_columns = raw_df.columns[raw_df.columns.str.contains('Ускорение|наклон', regex=True)]

    # Combine columns
    selected_columns = explicit_columns + list(pattern_columns) + [hole_col]

    new_names = ['lat', 'lon', 'vel', 'acc_X', 'acc_Y', 'acc_Z', 'fb_tilt',
                 'tilt', 'hole']
    # print(dict(zip(selected_columns, new_names)))
    names_map = {selected_columns[i]: new_names[i] for i in range(len(selected_columns))}
    try:
        # Filter and rename DataFrame
        filtered_df = raw_df[selected_columns]
    except Exception as e:
        print(f"Trouble with {path}:")
        print(e)
        return None
    df = filtered_df.rename(columns=names_map)

    df['hole'] = np.where(df['hole']>0, 1, 0)

    df['acc'] = calculate_summed_magnitude(df, 'acc_')

    return df

def read_raw_dirdata(dir_path: str, csv_pattern: str, func: Callable[[str], pd.DataFrame]=read_truck_data) -> List[pd.DataFrame]:
    csv_pattern+='.csv'
    pattern = re.compile(csv_pattern)
    try:
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if pattern.match(f)]
        return [func(file) for file in files]
    except:
        return []

def get_columns(folder_path: str, keep_latlon=False):
    file_path = glob.glob(f'{folder_path}/*.csv')[0]
    df = pd.read_csv(file_path, sep=';', dtype=np.float32)
    if not keep_latlon:
        df = df.drop(columns=['lat', 'lon'])
    return df.columns

def load_prepared(folder_path:str, keep_latlon=False, sample_frac=1)->pd.DataFrame:
    dataframes = []
    for filename in tqdm(os.listdir(folder_path), desc='Loading data'):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, sep=';', dtype=np.float32)
            if not keep_latlon:
                df = df.drop(columns=['lat', 'lon'])
            dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True).sample(frac=sample_frac)

def read_new_points(path: str) -> Optional[pd.DataFrame]:
    try:
        raw_df = pd.read_csv(path, delimiter=';', encoding='windows-1251', index_col=0)
    except:
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
    df['class'] = df['class'].str.replace(r'.*?(\d+).*', r'\1', regex=True)
    df['acc'] = calculate_summed_magnitude(df, 'acc_')

    df = convert_dash_to_nan(df)
    df['class'] = df['class'].fillna(0)
    return df