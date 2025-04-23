from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

from Transformer import RollingWindowTransformer
from exploration.data_read import read_dir_csvs

data_transformers = {ws:RollingWindowTransformer({
    'rot_X': ['', 'std', 'cv', 'iqr', 'skew', 'var'],
    'rot_Y': ['', 'std', 'cv', 'iqr', 'skew', 'var'],
    'acc_X': ['', 'std', 'kurt', 'var', 'iqr', ],
    'acc_Y': ['', 'std', 'kurt', 'var', 'iqr', ],
    'acc_X_zscore': ['', 'std', 'kurt', 'var', 'iqr', ],
    'acc_Y_zscore': ['', 'std', 'kurt', 'var', 'iqr', ],
    'acc_Z': ['', 'std', 'kurt', 'var','iqr', 'range'],
    'acc_Z_zscore': ['', 'std', 'kurt', 'var','iqr', 'range'],
    'acc': ['', 'std', 'var', 'iqr', 'kurt', 'cv', 'skew'],
    'acc_zscore': ['', 'std', 'var', 'iqr', 'kurt', 'cv', 'skew'],
}, window_size=ws) for ws in range(5, 60)}


def transform_data(
        tracks,
        transformer,
        output_folder,
        paths_func: Callable[[int], Path],
        dir_pattern=r'[0-9]{1,3}_w',
):
    """
    Preprocess data with flexible current_transformer, output folder, and dir reading function.

    Args:
        tracks: Iterable of track IDs (e.g., range(1, 39)).
        dir_path_func: Function to generate directory paths (e.g., lambda n: f"data/routes/route{n}").
        transformer: Transformer object (e.g., RollingWindowTransformer instance).
        output_folder: Path or string for output folder (e.g., "data/prepared10").
        read_dir_func: Function to read directory data (default: read_raw_dirdata).
        dir_pattern: Regex pattern for file matching (default: r'[0-9]{1,3}_w').
    """
    dir_names = [paths_func(i) for i in tracks]
    num_tracks = []

    # Process each directory
    for dir_name in tqdm(dir_names, desc=f"Transforming {paths_func(0).parent.name} with {transformer.window_size} rolling window"):
        combined_routes = read_dir_csvs(dir_name, pd.read_csv, dir_pattern)
        rolled_new_paths = [transformer.transform(df) for df in combined_routes if not df.empty]
        routeID = Path(dir_name).name
        (Path(output_folder) / f"{routeID}").mkdir(parents=True, exist_ok=True)
        num_tracks.append(len(rolled_new_paths))
        if rolled_new_paths:  # Only process if there’s at least one non-empty DataFrame

            for i, df in enumerate(rolled_new_paths):
                df.to_csv(Path(output_folder) / f"{routeID}/{i+1}_w.csv", index=False)

    print(f"Processed {len(num_tracks)} paths with {sum(num_tracks)} total tracks")

def roll_data(in_routes_folder: str|Path, ws:int):
    tracks = range(1, 39)
    in_routes_folder = Path(in_routes_folder)
    dir_path_func = lambda n: in_routes_folder / f"route{n}"
    output_folder = f"data/rolled/{in_routes_folder.name}_rolled{ws}"

    # Ensure output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Run preprocessing
    transform_data(
        tracks=tracks,
        transformer=data_transformers[ws],
        output_folder=output_folder,
        paths_func=dir_path_func,
    )


if __name__ == "__main__":
    for rws in [10]:
        roll_data(f'data/normalized/extremes_w10_norm', rws)