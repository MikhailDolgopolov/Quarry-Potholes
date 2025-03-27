import os
import pickle
import random

import numpy as np

import re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import Interval
from sklearn.utils._param_validation import RealNotInt


def calculate_summed_magnitude(df, prefix):
    # Select columns starting with the specified prefix
    pattern = f"{prefix}.*[XYZ]$"
    cols = [col for col in df.columns if re.match(pattern, col)]
    if len(cols) != 3:
        raise ValueError(f"Expected exactly 3 columns with prefix '{prefix}', but found {len(cols)}: {cols}")

    # Calculate the magnitude row-wise
    magnitude = np.sqrt((df[cols] ** 2).sum(axis=1))
    return magnitude


def convert_dash_to_nan(df):
    """
    Convert columns with '-' as missing values to numeric type in a pandas DataFrame.
    Only columns where all non-'-' values (after stripping whitespace) are numeric are converted.

    Parameters:
        df (pandas.DataFrame): Input DataFrame

    Returns:
        pandas.DataFrame: DataFrame with corrected numeric columns
    """
    for col in df.columns:
        # Check if the column is of object dtype (typically strings in pandas)
        if df[col].dtype == 'object':
            # Select values where, after stripping whitespace, the value is not '-'
            mask = df[col].str.strip() != '-'
            non_dash = df[col][mask]
            # Attempt to convert these non-'-' values to numeric, coercing errors to NaN
            converted = pd.to_numeric(non_dash, errors='coerce')
            # If all converted values are not NaN, the column is likely numeric with '-' as missing
            if converted.notna().all():
                # Replace values that strip to '-' with NaN
                df[col] = df[col].apply(lambda x: np.nan if x.strip() == '-' else x)
                # Convert the entire column to numeric
                df[col] = pd.to_numeric(df[col])
    return df

def train_split_by_column(df, y_column:str, test_frac: Interval(RealNotInt, 0, 1, closed="neither")):
    train_df, test_df = train_test_split(df, test_size=test_frac)
    X_train, y_train = train_df.drop(columns=[y_column]), train_df[y_column]

    X_test, y_test = test_df.drop(columns=[y_column]), test_df[y_column]
    return X_train, y_train, X_test, y_test

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def discretize_to_levels(arr, levels):
    """
    Round array values to the nearest specified levels

    Args:
        arr: Input array (any shape)
        levels: Array of target discrete values

    Returns:
        Discretized array with same shape as input
    """
    levels = np.asarray(levels)
    # Find closest level for each element
    indices = np.argmin(np.abs(arr[..., np.newaxis] - levels), axis=-1)
    return levels[indices]


def select_random_file(folder_path):
    """Select a random file from a folder"""
    try:
        # List all files in the folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        if not files:
            raise FileNotFoundError(f"No files found in {folder_path}")

        # Select a random file
        selected_file = random.choice(files)
        return os.path.join(folder_path, selected_file)
    except Exception as e:
        print(f"Error selecting file: {e}")
        return None


if __name__ == '__main__':
    import os

    directory = "models/LVQs"  # Change this to your actual directory
    suffix = "-resampled1.2"  # Change this to your desired suffix

    # for filename in os.listdir(directory):
    #     old_path = os.path.join(directory, filename)
    #
    #     if os.path.isfile(old_path):  # Ensure it's a file
    #         name, ext = os.path.splitext(filename)  # Separate name and extension
    #         new_filename = f"{name}{suffix}{ext}"  # Append suffix before extension
    #         new_path = os.path.join(directory, new_filename)
    #
    #         os.rename(old_path, new_path)
    #         print(f"Renamed: {filename} -> {new_filename}")