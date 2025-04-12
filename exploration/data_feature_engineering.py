from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

from Transformer import RollingWindowTransformer
from exploration.data_read import read_dir_csvs

data_transformers = {ws:RollingWindowTransformer({
    'rot_X': ['std', 'cv', 'iqr', 'skew', 'var'],
    'rot_Y': ['std', 'cv', 'iqr', 'skew', 'var'],
    'acc_X': ['std', 'kurt', 'var', 'iqr', ],
    'acc_Y': ['std', 'kurt', 'var', 'iqr', ],
    'acc_Z': ['std', 'kurt', 'var','iqr', 'range'],
    'acc': ['std', 'var', 'iqr', 'kurt', 'cv', 'skew'],
}, window_size=ws) for ws in range(5, 20)}


def transform_data(
        tracks,
        transformer,
        output_folder,
        paths_func: Callable[[int], str|Path],
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
    read_pretty_track = lambda x: pd.read_csv(x, sep=',')
    # Process each directory
    for dir_name in tqdm(dir_names, desc=f"Transforming {output_folder} with {transformer.window_size} rolling window"):
        # print(f"Processing {dir_name}")
        combined_routes = read_dir_csvs(dir_name, dir_pattern, read_pretty_track)
        # print(combined_routes[0])
        rolled_new_paths = [transformer.transform(df) for df in combined_routes if not df.empty]

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
    for route, df in tqdm(preprocessed_dfs.items()):
        output_path = Path(output_folder) / f"{route}.csv"
        df.to_csv(output_path, index=False)

    print(f"Processed {len(preprocessed_dfs)} paths with {sum(num_tracks)} total tracks")

def roll_data(routes_folder: str, ws:int):
    tracks = range(1, 36)
    dir_path_func = lambda n: Path('data/preprocessed') / Path(routes_folder) / f"route{n}"
    output_folder = f"data/engineered/{routes_folder}/rolled{ws}"
    # if dir_path_func(30).exists():
    #     return

    # Run preprocessing
    transform_data(
        tracks=tracks,
        transformer=data_transformers[ws],
        output_folder=output_folder,
        paths_func=dir_path_func,
    )


if __name__ == "__main__":
    roll_data('30peaks', 7)