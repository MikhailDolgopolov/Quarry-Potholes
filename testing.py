import pandas as pd
from data_read import read_raw_dirdata, read_new_points, load_prepared

df = load_prepared('data/class10')
print(df['class'].unique())