import os
import pickle
import copy
from pathlib import Path
from pprint import pprint, pformat
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import classification_report, f1_score, pairwise_distances
from sklearn.utils import resample
from sklvq import GLVQ

from evaluate.draw_functions import lvq_class_separation
from exploration.data_read import load_engineered_data, load_plain_data
from exploration.features.Separability import anomaly_features, pressure_features
from geospacial.map_drawings import parse_anomaly_parameters
from helpers import train_split_by_column


if __name__ == '__main__':

    def generate_model_filename(config):
        """Generate a filename based on the configuration."""
        return (
            f"glvq_hole{config['ws']}_[{'_'.join(map(str, config['glvq_params']['prototype_n_per_class']))}]_"
            f"{config['glvq_params']['solver_type']}_{config['glvq_params']['distance_type']}.pkl"
        )


    def get_model_predictions(
            config: dict,
            X_train,
            y_train,
            X_test,
            retrain=False,
            save_best=True,
            features: list[str] = None
    ) -> tuple[dict, np.ndarray, GLVQ]:
        """
        Get predictions from a GLVQ model, either by training a new model or loading an existing one.

        Parameters
        ----------
        config : dict
            GLVQ config including training options and GLVQ parameters.
        X_train, y_train : training data
        X_test : test features
        retrain : bool
            Whether to retrain the model even if a saved one exists.
        save_best : bool
            Whether to save the trained model.
        features : list[str], optional
            A list of features to select from X_train and X_test. If None, uses all features.
            """
        if features is not None:
            X_train = X_train[features]
            X_test = X_test[features]
        model_dir = 'models/LVQs'
        os.makedirs(model_dir, exist_ok=True)

        model_filename = generate_model_filename(config)
        model_path = Path(model_dir) / model_filename
        if not retrain and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return config, model.predict(X_test), model

        if config.get('use_resampling', False):
            # Use SMOTE for oversampling the minority class.
            # sampling_strategy (a float) indicates the desired ratio of minority to majority after resampling.
            smote = SMOTE(sampling_strategy=config.get('resampling_ratio', 1.0), random_state=42)
            X_train_processed, y_train_processed = smote.fit_resample(X_train, y_train)
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


    def lvq_distance_score(model, X):
        # choose the same metric GLVQ used
        metric = 'sqeuclidean' if model.distance_type.startswith('squared') else 'euclidean'

        P = model.prototypes_
        L = model.prototypes_labels_
        D = pairwise_distances(X, P, metric=metric)

        d_neg = D[:, L == 0].min(axis=1)
        d_pos = D[:, L == 1].min(axis=1)

        # “closeness” to positive class
        scores = d_neg / (d_neg + d_pos + 1e-12)
        return scores


    def tune_lvq_threshold(model, X, y_true):
        scores = lvq_distance_score(model, X)
        best_f1, best_t, best_report = 0, 0.5, None

        for t in np.linspace(0.0, 1.0, 101):
            y_pred = (scores >= t).astype(int)
            f = f1_score(y_true, y_pred, average='macro')
            if f > best_f1:
                best_f1, best_t = f, t
                best_report = classification_report(y_true, y_pred, output_dict=True)

        return best_t, best_f1, best_report

    df = load_plain_data('data/rolled/extremes_w10_norm_rolled10').sample(frac=0.2, random_state=42)
    print(df['severity'].value_counts(bins=3))
    def test_lvq(features):
        X_train, y_train, X_test, y_test = train_split_by_column(df, 'severity', 0.2)
        X_train, X_test = X_train[features], X_test[features]
        y_train, y_test = np.where(y_train>=0.3, 1, 0), np.where(y_test>=0.3, 1, 0)
        config = {
            'ws': 10,
            'use_resampling': True,
            'glvq_params': {
                'solver_type': 'sgd',
                'distance_type': 'squared-euclidean',
                'prototype_n_per_class': [2, 3],
            },
        }

        params, y_pred, model = get_model_predictions(
            config, X_train, y_train, X_test,
            retrain=True, save_best=False
        )
        print(pformat(features)[:40],'\n', classification_report(y_test, y_pred))

    print(len(df.columns))
    test_lvq(anomaly_features)
    print(len(df.columns))
    test_lvq(pressure_features)


    # 3b) tune threshold
    # t, f, report = tune_lvq_threshold(model, X_test, y_test)
    # print(f"Best threshold = {t:.2f}, macro‑F1 = {f:.3f}")
    # print(pd.DataFrame(report).T)
    #
    # # 3c) final preds at tuned threshold
    # scores = lvq_distance_score(model, X_test)
    # y_tuned = (scores >= t).astype(int)
    # print("Tuned:\n", classification_report(y_test, y_tuned))