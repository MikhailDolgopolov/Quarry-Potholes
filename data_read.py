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

def read_raw_dirdata(dir_path: str, csv_pattern: str) -> List[pd.DataFrame]:
    csv_pattern+='.csv'
    pattern = re.compile(csv_pattern)
    try:
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if pattern.match(f)]
        return [read_truck_data(file) for file in files]
    except:
        return []


def load_prepared(folder_path):
    # Step 2: Load all CSV files into a single DataFrame
    dataframes = []
    for filename in tqdm(os.listdir(folder_path), desc='Loading data'):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, sep=';', dtype=np.float32)
            df = df.drop(columns=['lat', 'lon'])
            dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)