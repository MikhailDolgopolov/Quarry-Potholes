from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from ComplexTransformer import MultiWindowRollingTransformer
from Transformer import RollingWindowTransformer
from exploration.data_read import read_truck_data, read_raw_dirdata, read_new_points

data_transformers = {ws:RollingWindowTransformer({
    'rot_X': ['std', 'cv', 'iqr', 'skew', 'var'],
    'rot_Y': ['std', 'cv', 'iqr', 'skew', 'var'],
    'acc_X': ['std', 'kurt', 'var', 'iqr', ],
    'acc_Y': ['std', 'kurt', 'var', 'iqr', ],
    'acc_Z': ['std', 'kurt', 'var','iqr', 'range'],
    'acc': ['std', 'var', 'iqr', 'kurt', 'cv', 'skew'],
}, window_size=ws) for ws in [5, 7, 10]}


def preprocess_data(
        tracks,
        transformer,
        output_folder,
        paths_func: Callable[[int], str],
        read_func=read_truck_data,
        dir_pattern=r'[0-9]{1,3}_w',
):
    """
    Preprocess data with flexible current_transformer, output folder, and dir reading function.

    Args:
        tracks: Iterable of track IDs (e.g., range(1, 36)).
        dir_path_func: Function to generate directory paths (e.g., lambda n: f"data/routes/route{n}").
        transformer: Transformer object (e.g., RollingWindowTransformer instance).
        output_folder: Path or string for output folder (e.g., "data/prepared10").
        read_dir_func: Function to read directory data (default: read_raw_dirdata).
        dir_pattern: Regex pattern for file matching (default: r'[0-9]{1,3}_w').
    """
    preprocessed_dfs = {}
    dir_names = [paths_func(i) for i in tracks]
    num_tracks = []

    # Process each directory
    mould = lambda x: transformer.transform(x) if transformer is not None else x
    for dir_name in tqdm(dir_names, desc="Processing paths"):
        new_path = read_raw_dirdata(dir_name, dir_pattern, read_func)
        rolled_new_paths = [mould(df) for df in new_path if not df.empty]

        routeID = Path(dir_name).name
        if rolled_new_paths:  # Only process if there’s at least one non-empty DataFrame
            num_tracks.append(len(rolled_new_paths))
            if len(rolled_new_paths) == 1:
                preprocessed_dfs[routeID] = rolled_new_paths[0]
            else:
                preprocessed_dfs[routeID] = pd.concat(rolled_new_paths, ignore_index=True)

    # Ensure output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Save processed DataFrames
    for route, df in tqdm(preprocessed_dfs.items(), desc="Saving data"):
        output_path = Path(output_folder) / f"{route}.csv"
        df.to_csv(output_path, index=False, sep=';')

    print(f"Processed {len(preprocessed_dfs)} paths with {sum(num_tracks)} total tracks")

def prepare_ws(target, ws):
    tracks = range(1, 36)
    dir_path_func = lambda n: f"data/routes/route{n}"

    output_folder = f"data/{target}{ws}"

    read_func = read_truck_data if target == 'hole' else read_new_points
    t = None if target == 'raw' else data_transformers[ws]
    # Run preprocessing
    preprocess_data(
        tracks=tracks,
        transformer=t,
        output_folder=output_folder,
        paths_func=dir_path_func,
        read_func=read_func,
    )

if __name__ == "__main__":
    variants = {
        "target": ["hole","class"],
        "ws": [5, 7, 10]
    }
    for combination in ParameterGrid(variants):
        prepare_ws(**combination)