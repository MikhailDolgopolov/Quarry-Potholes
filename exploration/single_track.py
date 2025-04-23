import os
import pickle
import random
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exploration.data_read import read_raw_track
from helpers import count_route_files, discretize_to_levels

# target, ws = 'severity', 10
# df_full = load_engineered_data(f'data/{target}{ws}', keep_latlon=True, sample_frac=1)
# print(len(df_full))
fig=plt.figure(figsize=(10, 6))
data_dir = Path(f'data/normalized/')
r = random.randint(1,38)
track_num = count_route_files(data_dir, r)
while track_num <2:
    r = random.randint(1, 38)
    track_num = count_route_files(data_dir, r)
track = random.randint(1, track_num)
print(f'track {track}')

# raw = read_raw_track(Path('data/input-raw')/f'route{r}/{track}_w.csv')
#
# print(raw.describe())
# aggregation = {
#     'front': ['pressure_FL', 'pressure_FR'],
#     'back': ['pressure_RL', 'pressure_RR'],
# }
# severity_columns = []
# pressure_columns = np.array([*aggregation.values()]).flatten()
# for part, columns in aggregation.items():
#     raw[part] = raw[columns].mean(axis=1)
for ews in 10, 20, 30:
    track_file = data_dir / f'extremes_w{ews}_norm/route{r}/{track}_w.csv'
    df = pd.read_csv(track_file)
    plt.plot(df['severity'], label=f'{ews} extremes', alpha=0.8)
# plt.plot(raw['front'], label=f'Front Pressure', alpha=0.8)
# plt.plot(raw['back'], label=f'Rear Pressure', alpha=0.8)
# plt.plot(raw['acc_Y'], label=f'Y acc', alpha=0.8)
# plt.xlim(50, 100)
plt.title(f'Route {r}, Track {track}')
plt.legend()
plt.show()
plt.close(fig)

