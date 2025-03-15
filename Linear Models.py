import pickle
from pprint import pprint

import numpy as np
from pygam import PoissonGAM
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, ParameterGrid
from tqdm import tqdm
import warnings
from data_read import load_prepared

# Suppress all warnings
warnings.filterwarnings("ignore")


def train_evaluate(params, X, y):
    """Train and evaluate in one step without validation"""
    try:
        model = PoissonGAM(lam=params['lam'], tol=params['tol'], n_splines=params['n_splines'], max_iter=50)
        model.fit(X, y)
        preds = model.predict(X)
        return mean_absolute_error(y, preds), params
    except:
        return np.inf, None


# Main execution
if __name__ == "__main__":
    # Simple grid: 2 values for lam × 2 values for tol = 4 combinations
    param_grid = {
        'lam': [30, 70, 100],
        'tol': [1e-2, 1e-1],
        'n_splines': [10, 15]
    }

    # Generate grid
    param_combinations = list(ParameterGrid(param_grid))

    print(f"Total combinations: {len(param_combinations)}")

    # Load data
    ws = 10
    target='class'
    big_df = load_prepared(f"data/class{ws}", sample_frac=0.5)

    train_df, test_df = train_test_split(big_df, test_size=0.5)

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    best_score = 100
    best_params = None

    results=dict()

    # Manual grid search
    for params in tqdm(param_combinations, desc="Testing parameters"):
        score, _ = train_evaluate(params, X_train, y_train)
        results[params]=score

        if score < best_score:
            best_score = score
            best_params = params

    # Final training with best params
    if best_params:
        print(f"Best parameters: {best_params}")
        print(f"Training MAE: {best_score:.2f}")

        final_model = PoissonGAM(**best_params, max_iter=200).fit(X_test, y_test)
        tMAE = mean_absolute_error(y_test, final_model.predict(X_test))
        print(f'Test MAE: {tMAE:.1f}')
        filename=f"models/simple_gam_{ws}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(final_model, f)

        results = dict(sorted(results.items(), key=lambda item: item[1]))[:len(param_combinations)//3]
        pprint(results)
    else:
        print("All parameter combinations failed")