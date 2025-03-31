import itertools
import os
import pickle
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklvq import GLVQ
from tqdm import tqdm

from evaluate.draw_functions import lvq_class_separation
from exploration.data_read import load_prepared
from helpers import train_split_by_column

def save_model(model, model_path):
    """Save model to disk using pickle."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(model_path):
    """Load model from disk."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return model


def get_model_predictions(config, X_train, y_train, X_test, retrain=False):
    """
    Get predictions from a GLVQ model for multiclass classification.
    It balances the class sizes via oversampling (if use_resampling is True)
    and then trains (or loads) the model accordingly.

    Args:
        config (dict): Configuration dictionary with parameters.
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.
        X_test (DataFrame): Testing features.
        retrain (bool): Force retraining even if model exists.

    Returns:
        tuple: (y_test, y_pred, model)
    """
    # Unpack configuration
    prototypes = config['prototypes']
    target = 'class'
    ws = config['ws']
    use_resampling = config['use_resampling']
    res_ratio = config.get('resampling_ratio', 1.0)

    # Create model paths
    prots = f'[{",".join(map(str, prototypes))}]'
    res_option = f'-resampled{res_ratio}' if use_resampling else ''
    model_dir = 'models/LVQs'
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(
        model_dir,
        f'glvq_{target}{ws}-{prots}{res_option}.pkl'
    )

    # Try to load existing model
    if not retrain and os.path.exists(model_path):
        model = load_model(model_path)
    else:
        # Handle class imbalance for multiclass
        if use_resampling:
            print(len(X_train))
            # Determine unique classes and find maximum sample count
            classes = np.unique(y_train)
            class_counts = y_train.value_counts()
            n_max = class_counts.max()
            X_list, y_list = [], []
            for c in classes:
                X_c = X_train[y_train == c]
                y_c = y_train[y_train == c]
                # Oversample to n_max * res_ratio samples for class c
                n_samples = int(n_max * res_ratio)
                X_c_res, y_c_res = resample(
                    X_c, y_c,
                    replace=True,
                    n_samples=n_samples,
                    random_state=42
                )
                X_list.append(X_c_res)
                y_list.append(y_c_res)
            X_train_processed = pd.concat(X_list)
            y_train_processed = pd.concat(y_list)
            print(len(X_train_processed))
        else:
            X_train_processed = X_train.copy()
            y_train_processed = y_train.copy()

        # Train and save model
        model = GLVQ(prototype_n_per_class=np.array(prototypes))
        model.fit(X_train_processed, y_train_processed)
        save_model(model, model_path)
        print(f"Model saved to {model_path}")

    # Generate predictions from X_test
    y_pred = model.predict(X_test)
    return y_test, y_pred, model


# Example usage:
if __name__ == '__main__':
    target, ws= 'class', 10
    df = load_prepared(f'data/{target}{ws}', sample_frac=0.2)
    X_train, y_train, X_test, y_test = train_split_by_column(df, target, 0.3)

    base_config = {
        'ws': ws,
        'use_resampling': True,
        'cols': ['acc_Z_std', 'acc_X_std', 'acc_X_var', 'acc_var', 'acc_std', 'acc_Z_range', 'acc_cv', 'acc_Z_iqr',
                 'acc_X_iqr'],
    }
    # Iterate over all combinations of prototypes and res_ratio
    config = base_config.copy()
    prototypes = np.full((6), 1)
    config['prototypes'] = prototypes
    y_p, y_t, model = get_model_predictions(config,
            X_train[config['cols']], y_train, X_test[config['cols']], False)

    if y_p is not None:
        print(classification_report(y_t, y_p))
        report = classification_report(y_t, y_p, output_dict=True)
        params = f"protos{','.join(map(str, prototypes))}"
        # img_path = f'images/LVQ_comparison/ws{ws}/{params}.png'
        # if report['accuracy'] >= 0.7:
        lvq_class_separation(model, model.prototypes_, config['cols'])