import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from Transformer import RollDataTransformer
from data_read import read_truck_data, read_dir_data

tracks = range(1, 36)


dir_path = lambda n: f"data/routes/route{n}"

ws=7
rdTrans = RollDataTransformer({
        'vel': ['', 'std'],
        'rot_X': ['', 'std', 'range'],
        'rot_Y': ['', 'std', 'range'],
        # 'acc_X': ['max', 'std'],
        'acc_Y': ['max', 'std'],
        # 'acc_Z': ['max', 'std',],
        'acc': ['', 'max', 'std', 'range'],
        'fb_tilt': ['max', 'std',],
        'tilt': ['max', 'std',],
        # 'jolt': ['max', 'std'],
        # 'jolt_Y': ['max', 'std'],
    },
        window_size=ws)
preprocessed_dfs=dict()
# Process directories with a progress bar
dir_names = [dir_path(i) for i in tracks]
num_tracks = []
for dir_name in tqdm(dir_names, desc="Processing paths"):
    new_path = read_dir_data(dir_name, r'[0-9]{1,3}_w')
    # Process dataframes within the directory with a progress bar
    rolled_new_paths = [rdTrans.roll_data(df) for df in new_path]
    routeID = dir_name.split(r'/')[-1]
    if len(rolled_new_paths)>0:
        num_tracks.append(len(rolled_new_paths))
    match len(rolled_new_paths):
        case 0:
            pass
        case 1:
            preprocessed_dfs[routeID]=rolled_new_paths[0]
        case _:
            preprocessed_dfs[routeID]=pd.concat(rolled_new_paths, ignore_index=True)

print(f"Loaded {len(preprocessed_dfs)} paths")
print(f"On average, they have {np.average(num_tracks):.0f} tracks per path. Max is {max(num_tracks)}")

Path(f'data/prepared{ws}').mkdir(parents=True, exist_ok=True)
for route, df in tqdm(preprocessed_dfs.items(), desc='saving data'):
    df.to_csv(f'data/prepared{ws}/{route}.csv', index=False, sep=';')
