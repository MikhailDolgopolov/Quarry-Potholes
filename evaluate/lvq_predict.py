import pickle
from pathlib import Path

import pandas as pd
from sklvq import GLVQ
from tqdm import tqdm

from exploration.data_read import read_dir_csvs
from exploration.features.Separability import anomaly_features

input_dir = Path('data/anomalies/threshold1.3_2.0_rolled10')
output_dir = Path('data/predicted/lvq')/input_dir.name
output_dir.mkdir(parents=True, exist_ok=True)
folders = [f for f in input_dir.iterdir() if 'route' in f.name and f.is_dir()]
with open('models/LVQs/glvq_hole10_[2_3]_sgd_squared-euclidean.pkl', 'rb') as f:
    lvq:GLVQ = pickle.load(f)
for route_folder in tqdm(folders, desc=f'Predicting with LVQ'):
    track_dfs = read_dir_csvs(route_folder, pd.read_csv)
    output_path = output_dir / route_folder.name
    output_path.mkdir(parents=True, exist_ok=True)

    for i, track_df in enumerate(track_dfs):
        if track_df.empty:
            continue
        track_df['pothole'] = lvq.predict(track_df[anomaly_features])
        track_df['severity'] = track_df['pothole'].copy()
        track_df.to_csv(output_path / f'{i + 1}_w.csv', index=False)