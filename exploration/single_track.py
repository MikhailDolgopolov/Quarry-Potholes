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






