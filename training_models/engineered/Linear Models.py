import os.path
import pickle
from pprint import pprint
from typing import Literal

import numpy as np
import pandas as pd
from pygam import PoissonGAM, LinearGAM, GammaGAM, InvGaussGAM, ExpectileGAM
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.utils import compute_sample_weight
from tqdm import tqdm
import warnings
from exploration.data_read import load_engineered_data
from joblib import Parallel, delayed  # Added for parallel processing

# Suppress all warnings
warnings.filterwarnings("ignore")


def GridSearch(gam_type:Literal['PoissonGAM', 'LinearGAM'], param_grid: dict, df: pd.DataFrame, test_frac=0.4, name='1'):
    param_combinations = list(ParameterGrid(param_grid))
    print(f"Total combinations: {len(param_combinations)}")

    def train_evaluate(params, X, y):
        """Train and evaluate in one step without validation"""
        try:
            model = globals()[gam_type](lam=params['lam'],
                                        tol=params['tol'],
                                        n_splines=params['n_splines'],
                                        max_iter=100)
            model.fit(X, y, compute_sample_weight('balanced', y))
            preds = model.predict(X)
            return mean_absolute_error(y, preds,  compute_sample_weight('balanced', y)), params
        except Exception as e:
            return np.inf, params  # Return params even on failure for tracking

    # Split data once upfront
    train_df, test_df = train_test_split(df, test_size=test_frac)
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]

    best_score = np.inf
    best_params = None
    results = {}

    # Parallel execution with GPU utilization
    with Parallel(n_jobs=8, backend='threading') as parallel:  # Use all 8 cores
        # Process in parallel with progress bar
        processed_results = parallel(
            delayed(train_evaluate)(params, X_train, y_train)
            for params in tqdm(param_combinations, desc="Testing parameters")
        )

        # Process results
        for score, params in processed_results:
            results[frozenset(params.items())] = score  # Use hashable key

            if score < best_score:
                best_score = score
                best_params = params
    # Final evaluation and saving
    if best_params and best_score < np.inf:
        print(f"\nBest parameters: {best_params}")
        print(f"Training MAE: {best_score:.2f}")

        final_model = (globals()[gam_type](**best_params, max_iter=150)
                       .fit(X_train, y_train,  compute_sample_weight('balanced', y_train)))
        tMAE = mean_absolute_error(y_test, final_model.predict(X_test), compute_sample_weight('balanced', y_train))
        print(f'Test MAE: {tMAE:.1f}')

        if name == '1': name = f'{tMAE:.1f}'
        pic_path = f"models/{gam_type}[lam{best_params['lam']}]-{name}.pkl"
        if not (name == '1' and os.path.exists(pic_path)):
            with open(pic_path, "wb") as f:
                pickle.dump(final_model, f)

            print(f" Saved '{pic_path}'")
        else:
            print("Haven't saved to not override. Change the name parameter")

        print("\nTop results:")
        pprint(sorted(results.items(), key=lambda x: x[1]))
    else:
        print("All parameter combinations failed")


# Main execution
if __name__ == "__main__":
    param_grid = {
        'lam': [1, 20, 40],
        'tol': [1e-2, 1e-3],
        'n_splines': [10]
    }

    # Load data
    ws = 7
    target = 'severity'
    big_df = load_engineered_data(f"data/class{ws}", sample_frac=1)
    # GridSearch('LinearGAM', param_grid, big_df, test_frac=0.2)

    train_df, test_df = train_test_split(big_df, test_size=0.3)
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]

    typ = "LinearGAM"
    sample_weights = compute_sample_weight('balanced', y_train)

    lam=1
    m = locals()[typ](lam=lam, n_splines=10, tol=1e-1).fit(X_train, y_train, sample_weights)

    y_pred = m.predict(X_test)

    test_weights = compute_sample_weight('balanced', y_test)

    weighted_mae = mean_absolute_error(y_test, y_pred, sample_weight=test_weights)
    print(weighted_mae)
    pic_path = f"models/{typ}[lam{lam}][ws{ws}]-balanced{round(weighted_mae)}.pkl"

    with open(pic_path, "wb") as f:
        pickle.dump(m, f)