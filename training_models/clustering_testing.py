import pickle
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.utils import compute_sample_weight

from exploration.data_read import load_prepared
from geospacial.load_latlon import filter_reliable_potholes
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor

from helpers import train_split_by_column
from training_models.GradBoosting import train_evaluate


def clustering_grid_search(df: pd.DataFrame, param_grid, n_jobs=4):
    param_combinations = list(ParameterGrid(param_grid))
    print(f"Total combinations: {len(param_combinations)}")

    def evaluate_clustering(in_df, params):
        df_reliable = filter_reliable_potholes(in_df, **params, reliable_col=col)
        df_reliable = df_reliable[df_reliable[col]]
        df_reliable = df_reliable.drop(columns=[col, 'lat', 'lon'], errors='ignore')
        result_clustered_data = train_evaluate(None, *train_split_by_column(df_reliable, 'class', 0.2))
        result_raw = train_evaluate(None, *train_split_by_column(in_df, 'class', 0.2))

        # print(result_clustered_data)
        return {
            'params': params,
            'score': -result_clustered_data['test_mae']+result_raw['test_mae'],
            'model': result_clustered_data['model']
        }
    # # Run parallel evaluations
    # results = []
    # print(df.columns)

    with Parallel(n_jobs=n_jobs, verbose=10) as parallel:
        jobs = (delayed(evaluate_clustering)(df, params) for params in param_combinations)
        results = [result for result in parallel(jobs) if result]

    # Sort results by test MAE
    sorted_results = sorted(results, key=lambda x: x['score'])
    pprint(sorted_results[:5])
    return sorted_results[0]

if __name__ == '__main__':
    target = 'class'
    df = load_prepared(f'data/{target}10', keep_latlon=True, sample_frac=0.3)

    col = 'rel'
    grid = {
        'hole_threshold': [25, 30],
        'eps': [0.02, 0.03, 0.035],
        'cluster_samples': [5, 10, 15, 20],
        'reports': [10, 20, 30],
        'positive_class_ratio': [0.5, 0.6, 0.7, 0.8]
    }
    best_result = clustering_grid_search(df, grid)
    model = best_result['model']
    # print(type(model))
    # with open('models/clustering.pkl', 'wb') as fn:
    #     pickle.dump(model, fn)
