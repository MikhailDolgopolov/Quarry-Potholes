import pandas as pd
from data_read import read_raw_dirdata, read_new_points

df = read_raw_dirdata('data/routes/route3', r'[0-9]{1,3}_w', read_new_points)
# print(df)
print(df[0].columns)