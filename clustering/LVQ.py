import itertools
import os
import pickle
import copy
from pathlib import Path
from pprint import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
from sklvq import GLVQ
from sklearn.model_selection import ParameterGrid

from evaluate.draw_functions import LVQ_class_separation
from exploration.data_read import load_prepared
from helpers import train_split_by_column

features_for_LVQ = ['acc_Z_std', 'acc_X_std', 'acc_X_var', 'acc_var', 'acc_std', 'acc_Z_range', 'acc_cv', 'acc_Z_iqr',
                    'acc_X_iqr']

def predict_with_LVQ(model_path, X) -> np.ndarray:
    """Load a saved GLVQ model and predict using predefined LVQ columns."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model.predict(X[features_for_LVQ])

if __name__ == '__main__':

    def generate_model_filename(config):
        """Generate a filename based on the configuration."""
        return (
            f"glvq_hole{config['ws']}_[{'_'.join(map(str, config['prototypes']))}]_"
            f"{config['glvq_params']['solver_type']}_{config['glvq_params']['distance_type']}.pkl"
        )


    def get_model_predictions(config: dict, X_train, y_train, X_test, retrain=False, save_best=True)\
            -> tuple[dict, np.ndarray, GLVQ]:
        """Get predictions from a GLVQ model, either by training a new model or loading an existing one."""
        X_train, X_test = X_train[features_for_LVQ], X_test[features_for_LVQ]
        model_dir = 'models/LVQs'
        os.makedirs(model_dir, exist_ok=True)

        model_filename = generate_model_filename(config)
        model_path = Path(model_dir) / model_filename
        if not retrain and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return config, model.predict(X_test), model

        if config.get('use_resampling', False):
            minority_class = 1
            X_minority = X_train[y_train == minority_class]
            y_minority = y_train[y_train == minority_class]
            X_majority = X_train[y_train == 0]
            y_majority = y_train[y_train == 0]

            X_minority_oversampled, y_minority_oversampled = resample(
                X_minority, y_minority,
                replace=True,
                n_samples=int(len(X_majority) * config['resampling_ratio']),
                random_state=42
            )

            X_train_processed = pd.concat([X_majority, X_minority_oversampled])
            y_train_processed = pd.concat([y_majority, y_minority_oversampled])
        else:
            X_train_processed = X_train.copy()
            y_train_processed = y_train.copy()

        model = GLVQ(**config['glvq_params'])
        model.fit(X_train_processed, y_train_processed)
        predictions = model.predict(X_test)

        if save_best:
            best_model_path = os.path.join(model_dir, model_filename)
            with open(best_model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved to {best_model_path}")

        return config, predictions, model


    def gridsearch_glvq(X_train, y_train, X_test, y_test, base_config: dict, param_grid: dict, n_jobs=8):
        """Perform grid search for the best GLVQ model."""
        results = []
        best_acc = 0.0
        best_config = None
        best_model = None

        grid = list(ParameterGrid(param_grid))
        print(f"Total parameter combinations: {len(grid)}")

        search_results = Parallel(n_jobs=n_jobs)(
            delayed(get_model_predictions)(copy.deepcopy(base_config) | params, X_train, y_train, X_test, retrain=True)
            for params in tqdm(grid, desc="Gridsearch")
        )

        for config, acc, model in search_results:
            results.append({'config': config, 'accuracy': acc})
            if acc > best_acc:
                best_acc = acc
                best_config = config
                best_model = model

        # Save best model with its configuration
        if best_model:
            model_filename = generate_model_filename(best_config)
            model_path = os.path.join('models/LVQs', model_filename)
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            print(f"Best model saved to {model_path}")

        return best_config, best_model, results


    ws = 5
    df = load_prepared(f'data/hole{ws}', sample_frac=1)
    X_train, y_train, X_test, y_test = train_split_by_column(df, 'hole', 0.9)

    # Base configuration
    base_config = {
        'ws': ws,
        'use_resampling': True,
        'cols': features_for_LVQ,
        'resampling_ratio': 1.0,
        'glvq_params': {
            'solver_type': 'sgd',
            'distance_type': 'euclidean',
        },
        'prototypes': [2,2]
    }
    param_grid = {}


    config, y_pred, model = get_model_predictions(base_config, X_train, y_train, X_test)



