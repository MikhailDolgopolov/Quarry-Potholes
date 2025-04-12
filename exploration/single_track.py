import os
import random

import matplotlib.pyplot as plt
import pandas as pd

from exploration.data_read import read_raw_track
from helpers import select_random_file

# target, ws = 'severity', 10
# df_full = load_prepared(f'data/{target}{ws}', keep_latlon=True, sample_frac=1)
# print(len(df_full))

r = random.randint(1,35)
f=plt.figure(figsize=(10, 6))
l = len(os.listdir(f'data/preprocessed/10peaks/route{r}/'))
track = random.randint(1, l)
for p in [10, 30, 50]:
    track_file=f'data/preprocessed/{p}peaks/route{r}/{track}_w.csv'
    df = pd.read_csv(track_file)
    plt.plot(df['severity'], label=f'{p} peaks', alpha=0.9)
plt.legend()
plt.show()
plt.close(f)
# plt.figure(figsize=(10, 6))
# plt.plot([0] * len(df),  '--', color='gray',)
# plt.plot(df['acc_X'], label='X', alpha=0.8)
# plt.plot(df['acc_Y'], label='Y', alpha=0.8)
# plt.plot(df['acc_Z']*10, label='10х Acceleration Z', alpha=0.8)


# Add labels and title
# plt.xlabel('1Hz readings')
# plt.ylabel('Acceleration (m/s2)')
# plt.title('Accelerometers')
# plt.legend()
# plt.show()





