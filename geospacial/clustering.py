import os
from pathlib import Path

import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from tqdm import tqdm

from exploration.data_read import read_dir_csvs


def filter_potholes(input_folder: str | Path, route: int, eps: float, output_folder: str | Path):
    # Read all CSV files from the route folder and add a file id to each
    route_path = Path(input_folder) / f'route{route}'
    dfs = read_dir_csvs(route_path, pd.read_csv)
    if not dfs:
        # print(f"No files found in {route_path}")
        return

    # Concatenate with keys to record which file each row came from.
    # The keys (0, 1, 2, ...) serve as file_id.
    combined = pd.concat(dfs, keys=range(len(dfs)))
    combined = combined.reset_index(level=0).rename(columns={'level_0': 'file_id'})

    # Extract potholes (where 'pothole' equals 1)
    potholes = combined[combined['pothole'] == 1].copy()
    if potholes.empty:
        # print(f"No potholes found for route {route}")
        return

    # Define min_samples as a function of number of tracks.
    num_tracks = len(dfs)
    # For example, we use the floor of the square root of the number of tracks,
    # clipped between 1 and 10.
    min_samples = np.clip(np.floor(np.sqrt(num_tracks)), 2, 5).astype(int)

    # Run DBSCAN clustering on potholes (using lat/lon coordinates)
    coords = potholes[['lat', 'lon']].to_numpy()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(coords)
    potholes['cluster'] = db.labels_
    # Debug print: cluster sizes
    # print(potholes['cluster'].value_counts())

    reliable_indices = potholes[potholes['cluster'] != -1].index

    unreliable_mask = (combined['pothole'] == 1) & (~combined.index.isin(reliable_indices))
    combined.loc[unreliable_mask, 'pothole'] = 0
    combined.loc[unreliable_mask, 'severity'] = 0

    # Save each track (i.e., each file's data) individually.
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    os.makedirs(output_path / f'route{route}', exist_ok=True)
    for file_id, file_df in combined.groupby('file_id'):
        # Optionally drop the file_id column for output cleanliness.
        file_df = file_df.drop(columns=['file_id'])
        output_file = output_path / f'route{route}/{file_id+1}_w.csv'
        file_df.to_csv(output_file, index=False)
        # print(f"Saved filtered track {file_id} to {output_file}")

def cluster_routes(ws, e):
    input_folder = f'data/relabeled/ws{ws}_peaks'
    routes_range = range(1, 36)
    eps = e / 111000  # Convert 8 meters to degrees (approx.)
    for route in tqdm(routes_range, desc=f"Clustering of {e} meters"):
        filter_potholes(input_folder, route, eps, f'data/clustered/{ws}peaks_eps{e}')

if __name__ == "__main__":
    ws = 30
    for e in [2, 3, 5]:
        cluster_routes(ws, e)
