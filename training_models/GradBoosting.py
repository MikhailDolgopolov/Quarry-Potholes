from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import compute_sample_weight
import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from pprint import pprint

from data_read import load_prepared


def hgbr_grid_search(param_grid: dict, df: pd.DataFrame, target: str, test_frac=0.2, n_jobs=8):
    """
    Custom grid search for HistGradientBoostingRegressor
    """
    param_combinations = list(ParameterGrid(param_grid))
    print(f"Total combinations: {len(param_combinations)}")

    # Split data once upfront
    train_df, test_df = train_test_split(df, test_size=test_frac)
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]

    # Precompute sample weights
    train_weights = compute_sample_weight('balanced', y_train)
    test_weights = compute_sample_weight('balanced', y_test)

    def train_evaluate(params):
        """Train and evaluate a single model"""
        try:
            model = HistGradientBoostingRegressor(
                **params,
                max_iter=200,
                min_samples_leaf=5,
                # max_features=0.9,
                learning_rate=0.6,
                scoring='neg_mean_absolute_error',
                random_state=42
            )
            model.fit(X_train, y_train, sample_weight=train_weights)

            # Get validation score during training
            train_pred = model.predict(X_train)
            train_mse = mean_squared_error(y_train, train_pred, sample_weight=train_weights)

            # Test evaluation
            test_pred = model.predict(X_test)
            test_mae = mean_absolute_error(y_test, test_pred, sample_weight=test_weights)

            return {
                'params': params,
                'train_mse': train_mse,
                'test_mae': test_mae,
                'model': model
            }
        except Exception as e:
            print(f"Error with params {params}: {str(e)}")
            return None

    # Run parallel evaluations
    results = []

    with Parallel(n_jobs=n_jobs, verbose=10) as parallel:
        jobs = (delayed(train_evaluate)(params) for params in param_combinations)
        results = [result for result in parallel(jobs) if result]

    # Sort results by test MAE
    sorted_results = sorted(results, key=lambda x: x['train_mse'])

    # Save top 3 models
    for idx, result in enumerate(sorted_results[:3], 1):
        model = result['model']
        pstr = ''.join([f'[{k}{v}]' for k, v in result['params'].items()])
        model_name = f"HGBR_{pstr}_top{idx}_{round(result['test_mae'])}.pkl"

        with open(f"models/{model_name}", "wb") as f:
            pickle.dump(model, f)
    pprint(sorted_results[:3])
    return sorted_results

# Usage example
if __name__ == "__main__":
    pd.set_option('display.max_columns', 25)
    pd.set_option('display.width', 1000)

    ws = 10
    target = 'class'
    big_df = load_prepared(f"data/class{ws}", sample_frac=1)

    # Define parameter grid for boosting
    # param_grid = {
    #     'learning_rate': [0.05, 0.1, 0.2],
    #     'max_depth': [3, 5, 7],
    #     'min_samples_leaf': [20, 50, 100],
    #     'l2_regularization': [0.0, 0.1, 0.3],
    #     'loss': ['absolute_error', 'squared_error']
    # }

    param_grid = {
        'max_depth': [10, 12, 14, 16, None],
        'max_features': [1.0, 0.9],
        'l2_regularization': [0.2, 0.3, 0.4, 0.5, 0.6],

    }

    # Run grid search
    results = hgbr_grid_search(
        param_grid=param_grid,
        df=big_df,
        target=target,
        test_frac=0.1,
        n_jobs=8
    )