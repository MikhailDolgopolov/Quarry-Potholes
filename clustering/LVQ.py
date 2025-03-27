import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklvq import GLVQ

from evaluate.draw_functions import LVQ_class_separation
from exploration.data_read import load_prepared
from helpers import train_split_by_column

def save_model(model, scaler, model_path, scaler_path):
    """Save model and scaler to disk using pickle."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    if scaler:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

def load_model(model_path, scaler_path):
    """Load model and scaler from disk."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    return model, scaler


def get_model_predictions(config, retrain=False):
    """
    Get predictions from a GLVQ model based on configuration parameters.
    Trains and saves new model if none exists (with user confirmation).

    Args:
        config (dict): Configuration dictionary with parameters
        retrain (bool): Force retraining even if model exists

    Returns:
        tuple: (y_pred, y_test, model, scaler) or None if aborted
    """
    # Unpack configuration
    prototypes = config['prototypes']
    target = 'hole'
    ws = config['ws']
    use_scaler = config['use_scaler']
    use_resampling = config['use_resampling']
    res_ratio = config['res_ratio']
    imp_cols = config['imp_cols']
    sample_frac = config['sample_frac']
    test_size = config['test_size']

    # Create model paths
    prots = f'[{",".join(map(str, prototypes))}]'
    res_option = f'-resampled{res_ratio}' if use_resampling else ''
    scaled = 'scaled' if use_scaler else 'original'

    model_dir = 'models/LVQs'
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(
        model_dir,
        f'glvq_{target}{ws}-{scaled}-{prots}{res_option}.pkl'
    )
    scaler_path = os.path.join(
        model_dir,
        f'scaler_{target}{ws}-{scaled}-{prots}{res_option}.pkl'
    )

    # Try to load existing model
    if not retrain and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model, scaler = load_model(model_path, scaler_path)
        data_loaded = False
    else:
        # User confirmation for training
        if not retrain:
            response = input(f"No model found at {model_path}. Train new model? [y/N]: ").strip().lower()
            if response != 'y':
                print("Model training aborted by user")
                return None, None, None, None

        # Load and prepare data
        print("Loading and preparing data...")
        df = load_prepared(f'data/{target}{ws}', x_selection=imp_cols, sample_frac=sample_frac)
        X_train, y_train, X_test, y_test = train_split_by_column(df, target, test_size)

        # Handle class imbalance
        if use_resampling:
            minority_class = 1
            X_minority = X_train[y_train == minority_class]
            y_minority = y_train[y_train == minority_class]
            X_majority = X_train[y_train == 0]
            y_majority = y_train[y_train == 0]

            X_minority_oversampled, y_minority_oversampled = resample(
                X_minority, y_minority,
                replace=True,
                n_samples=int(len(X_majority) * res_ratio),
                random_state=42
            )

            X_train_processed = pd.concat([X_majority, X_minority_oversampled])
            y_train_processed = pd.concat([y_majority, y_minority_oversampled])
        else:
            X_train_processed = X_train.copy()
            y_train_processed = y_train.copy()

        # Feature scaling
        scaler = StandardScaler() if use_scaler else None
        if scaler:
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train_processed),
                columns=X_train_processed.columns
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns
            )
        else:
            X_train_scaled = X_train_processed.copy()
            X_test_scaled = X_test.copy()

        # Train and save model
        print("Training new model...")
        model = GLVQ(prototype_n_per_class=np.array(prototypes))
        model.fit(X_train_scaled, y_train_processed)
        save_model(model, scaler, model_path, scaler_path)
        print(f"Model saved to {model_path}")
        data_loaded = True

    # Generate predictions
    if not data_loaded:
        df = load_prepared(f'data/{target}{ws}', x_selection=imp_cols, sample_frac=sample_frac)
        _, _, X_test, y_test = train_split_by_column(df, target, test_size)

        if scaler:
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns
            )
        else:
            X_test_scaled = X_test.copy()

    y_pred = model.predict(X_test_scaled)
    return y_pred, y_test, model, scaler


# Example usage:
if __name__ == '__main__':
    config = {
        'prototypes': [2, 2],
        'ws': 7,
        'use_scaler': False,
        'use_resampling': True,
        'res_ratio': 1.2,
        'imp_cols': ['acc_X_std', 'acc_X_kurt', 'acc_X_var', 'acc_X_iqr',
                     'acc_Y_var', 'acc_Y_iqr', 'acc_std', 'acc_var', 'acc_iqr', 'acc_kurt'],
        'sample_frac': 0.3,
        'test_size': 0.2
    }

    y_pred, y_test, model, scaler = get_model_predictions(config)

    if y_pred is not None:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        # LVQ_class_separation(model, model.prototypes_, config['imp_cols'])