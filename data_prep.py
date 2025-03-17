from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

# Assuming these are defined elsewhere
from Transformer import RollingWindowTransformer
from data_read import read_raw_dirdata, read_truck_data, read_new_points


def add_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Add statistical features to the DataFrame."""
    frame['energy_proxy'] = frame['vel'] ** 2 + frame['acc'] ** 2
    return frame


def preprocess_data(
        tracks,
        transformer,
        output_folder,
        paths_func: Callable[[int], str],
        read_func=read_truck_data,
        dir_pattern=r'[0-9]{1,3}_w',
):
    """
    Preprocess data with flexible transformer, output folder, and dir reading function.

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
    for dir_name in tqdm(dir_names, desc="Processing paths"):
        new_path = read_raw_dirdata(dir_name, dir_pattern, read_func)
        rolled_new_paths = [transformer.roll_data(df) for df in new_path if not df.empty]

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


if __name__ == "__main__":
    tracks = range(1, 36)
    dir_path_func = lambda n: f"data/routes/route{n}"
    target, ws = 'class', 10
    transformer = RollingWindowTransformer({
        'rot_X': ['', 'var'],
        'rot_Y': ['', 'var', ],
        'acc_X': ['', 'std', 'max'],
        'acc_Z': ['', 'std', 'max'],
        'acc': ['', 'std', 'var', 'max', 'range'],
        'fb_tilt': ['max', 'var', 'range'],
        'tilt': ['max', 'var', 'range'],
    }, window_size=ws)
    output_folder = f"data/{target}{ws}"

    read_func = read_new_points if target=='class' else read_truck_data
    # Run preprocessing
    preprocess_data(
        tracks=tracks,
        transformer=transformer,
        output_folder=output_folder,
        paths_func=dir_path_func,
        read_func=read_func,
    )