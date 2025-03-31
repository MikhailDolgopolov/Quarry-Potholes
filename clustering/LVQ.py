import os
import pickle
import copy
from pathlib import Path
from pprint import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import resample
from sklvq import GLVQ
from sklearn.model_selection import ParameterGrid

from evaluate.draw_functions import lvq_class_separation
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
            f"glvq_hole{config['ws']}_[{'_'.join(map(str, config['glvq_params']['prototype_n_per_class']))}]_"
            f"{config['glvq_params']['solver_type']}_{config['glvq_params']['distance_type']}.pkl"
        )


    def get_model_predictions(config: dict, X_train, y_train, X_test, retrain=False, save_best=True) \
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


    def get_solver_options_grid() -> list:
        """Return a parameter grid with solver options and related parameters for GLVQ grid search."""
        solver_types = ['sgd', 'adam', 'bfgs']  # Available solver options from sklvq
        distance_types = ['euclidean', 'squared-euclidean']
        prototype_configs = [[2, 3], [2, 2], [3, 2], [3,3]]

        # Create a list of dictionaries for each combination
        param_grid = [
            {
                'glvq_params': {'solver_type': solver,
                                'distance_type': distance,
                                'prototype_n_per_class': prot},
            }
            for solver in solver_types
            for distance in distance_types
            for prot in prototype_configs
        ]
        return param_grid


    def gridsearch_glvq(X_train, y_train, X_test, y_test, base_config: dict, param_grid: list, n_jobs=4):
        """Perform grid search for the best GLVQ model based on F1 score for the positive class."""
        results = []
        best_f1 = 0.0
        best_config = None
        best_model = None

        print(f"Total parameter combinations: {len(param_grid)}")

        search_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(get_model_predictions)(copy.deepcopy(base_config) | params, X_train, y_train, X_test, retrain=True,
                                           save_best=False)
            for params in tqdm(param_grid, desc="Gridsearch")
        )

        for config, predictions, model in search_results:
            f1 = f1_score(y_test, predictions, pos_label=1)
            results.append({'config': config, 'f1_score': f1})
            if f1 > best_f1:
                best_f1 = f1
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

    def big_search():
        for ws in [7 ,10]:
            df = load_prepared(f'data/hole{ws}', sample_frac=0.8)
            X_train, y_train, X_test, y_test = train_split_by_column(df, 'hole', 0.8)

            # Base configuration
            base_config = {
                'ws': ws,
                'use_resampling': True,
                'cols': features_for_LVQ,
                'resampling_ratio': 1.0,
            }

            # Get parameter grid with solver options
            param_grid = get_solver_options_grid()

            # Perform grid search
            best_config, best_model, results = gridsearch_glvq(X_train, y_train, X_test, y_test, base_config, param_grid)

            # Print best configuration and its performance
            print(f"Best Configuration for ws {ws}:")
            pprint(best_config)
            y_pred_best = best_model.predict(X_test[features_for_LVQ])
            print(f"Classification Report for Best {ws} Model:")
            print(classification_report(y_test, y_pred_best))

            # Show top results
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('f1_score', ascending=False)
            print(f"Top 3 Configurations by F1 Score for {ws}:")
            print(results_df.head(3))

    # big_search()
    ws=10
    df = load_prepared(f'data/hole{ws}', sample_frac=1)
    X, y = df[features_for_LVQ], df['hole']
    model_path = 'models/LVQs/glvq_hole10_[3_3]_adam_squared-euclidean.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    #
    lvq_class_separation(model, X, features_for_LVQ)