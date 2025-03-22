import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import compute_sample_weight
import pandas as pd
import pickle
from joblib import Parallel, delayed
from pprint import pprint

from exploration.data_read import load_prepared


def train_evaluate(params, X_t, y_t, X_e, y_e, train_weights=None, eval_weights=None):
    if train_weights is None:
        train_weights = compute_sample_weight('balanced', y_t)
    if eval_weights is None:
        eval_weights = compute_sample_weight('balanced', y_e)
    if params is None:
        params = dict()

    try:
        # Get default parameters for comparison
        default_params = HistGradientBoostingRegressor().get_params()

        # Create model with merged parameters
        model = HistGradientBoostingRegressor(
            **params,
            max_iter=100,
            # learning_rate=0.6,
            scoring='neg_mean_absolute_error',
            random_state=42
        )
        model.fit(X_t, y_t, sample_weight=train_weights)

        # Identify non-default parameters
        non_default_params = {
            k: v for k, v in params.items()
            if str(v) != str(default_params.get(k, None))
        }

        # Evaluation metrics
        test_pred = model.predict(X_e)
        test_rmse = np.sqrt(mean_squared_error(y_e, test_pred, sample_weight=eval_weights))

        return {
            'params': non_default_params,  # Return only non-default params
            'test_rmse': test_rmse,
            'model': model
        }
    except Exception as e:
        print(f"Error with params {params}: {str(e)}")
        return None


def hgbr_grid_search(param_grid: dict, df: pd.DataFrame, target: str, test_frac=0.2, n_jobs=8):
    """
    Custom grid search for HistGradientBoostingRegressor
    """
    param_combinations = list(ParameterGrid(param_grid))
    print(f"Grid search combinations: {len(param_combinations)}")

    # Split data once upfront
    train_df, test_df = train_test_split(df, test_size=test_frac)
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]

    train_weights = compute_sample_weight('balanced', y_train)
    test_weights = compute_sample_weight('balanced', y_test)

    with Parallel(n_jobs=n_jobs, verbose=10) as parallel:
        jobs = (delayed(train_evaluate)(params, X_train, y_train, X_test, y_test, train_weights, test_weights) for params in param_combinations)
        results = [result for result in parallel(jobs) if result]

    # Sort results by test MAE
    sorted_results = sorted(results, key=lambda x: x['test_rmse'])

    # Save top 3 models
    for idx, result in enumerate(sorted_results[:3], 1):
        model = result['model']
        pstr = ''.join([f'[{k}{v}]' for k, v in result['params'].items()])
        model_name = f"HGBR_{pstr}_top{idx}_{round(result['test_rmse'])}.pkl"

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
    big_df = load_prepared(f"data/{target}{ws}", sample_frac=0.5)

    # Define parameter grid for boosting
    # param_grid = {
    #     'learning_rate': [0.05, 0.1, 0.2],
    #     'max_depth': [3, 5, 7],
    #     'min_samples_leaf': [20, 50, 100],
    #     'l2_regularization': [0.0, 0.1, 0.3],
    #     'loss': ['absolute_error', 'squared_error']
    # }

    param_grid = {
        # 'max_depth': [12, 14, None],
        'tol': [1e-2, 1e-4, 0.1],
        # 'max_features': [1.0, 0.9],
        'learning_rate': [0.2, 0.1, 1],
        'l2_regularization': [0.3, 0.1, 0.0,],
        'min_samples_leaf': [5, 20, 40],
        'loss': ['absolute_error', 'squared_error'],
    }

    # Run grid search
    results = hgbr_grid_search(
        param_grid=param_grid,
        df=big_df,
        target=target,
        test_frac=0.3,
        n_jobs=4
    )