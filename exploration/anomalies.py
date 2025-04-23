import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from tqdm import tqdm

from evaluate.draw_functions import plot_feature_ranges
from exploration.data_read import load_engineered_data, load_plain_data, read_dir_csvs
from helpers import train_split_by_column, split_off_target_cols


def isoforest_pothole_anomalies(ws:int, contamination_bias: float):
    target = 'severity'
    folder = f'data/rolled/extremes_w10_norm_rolled{ws}'
    contamination_bias = float(contamination_bias)
    big_df = load_plain_data(folder)
    X, y = split_off_target_cols(big_df.copy(), target)
    X = X.drop(columns=['lat', 'lon'])
    print(y.value_counts(bins=5))

    pothole_percentage = len(y[y > 0.3]) / len(y)*contamination_bias
    isoforest = IsolationForest(n_estimators=100, contamination=pothole_percentage, random_state=42)
    isoforest.fit(X)
    output_folder = Path(f'data/isoforest/{contamination_bias}_rolled{ws}')
    output_folder.mkdir(parents=True, exist_ok=True)
    first = ['severity', 'lat', 'lon']
    for route_folder in tqdm(list(Path(folder).iterdir())):
        track_dfs = read_dir_csvs(route_folder, pd.read_csv)
        output_path = output_folder/route_folder.name
        output_path.mkdir(parents=True, exist_ok=True)
        for i, track_df in enumerate(track_dfs):
            if len(track_df) == 0: continue
            x = track_df.drop(columns=['lat', 'lon', 'pothole', 'severity'], errors='ignore')
            track_df[target] = np.where(isoforest.predict(x) == -1, 1, 0)
            track_df[~track_df[target] == 0]['severity'] = 0
            order = first + [col for col in track_df.columns if col not in first]
            track_df[order].to_csv(output_path/f'{i+1}_w.csv', index=False)


if __name__ == '__main__':

    d = {
        'ws':[10],
        'contamination_bias':[1.5, 2.0]
    }
    for params in ParameterGrid(d):
        isoforest_pothole_anomalies(**params)

    # anomaly_labels = load_plain_data('data/anomalies/threshold1.3_1.0_rolled10')
    # original_labels = load_plain_data('data/rolled/threshold1.3_rolled10')
    #
    # plot_feature_ranges(original_labels, f=features)
    # plot_feature_ranges(anomaly_labels, f=features)