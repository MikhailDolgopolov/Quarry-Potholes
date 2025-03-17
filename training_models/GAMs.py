import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pygam import LogisticGAM
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy.special import expit
from data_read import load_prepared

def get_model(ws: int, X_train: pd.DataFrame, y_train: pd.Series, train_new=False) -> LogisticGAM:
    """
    Returns a fitted LogisticGAM model for binary classification of 'hole'.

    Parameters:
    - ws (int): Window size for the model filename.
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Binary target variable (0/1 for no hole/hole).
    - train_new (bool): If True, forces retraining of the model.

    Returns:
    - LogisticGAM: Fitted model for binary prediction.
    """
    filename = f"models/gam_hole{ws}.pkl"
    model = LogisticGAM(n_splines=15, lam=0.6, max_iter=50, tol=0.01)

    if train_new or not os.path.exists(filename):
        print("Fitting a LogisticGAM for hole prediction...")
        model.fit(X_train, y_train)
        with open(filename, "wb") as f:
            pickle.dump(model, f)
    else:
        try:
            with open(filename, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {filename}: {e}")

    return model

def draw_partial_deps(m: LogisticGAM, X_train: pd.DataFrame):
    """
    Draws partial dependence plots for each feature on the probability scale.

    Parameters:
    - m (LogisticGAM): Fitted LogisticGAM model.
    - X_train (pd.DataFrame): Training features for plotting.
    """
    feature_names = X_train.columns
    n_features = len(feature_names)
    n_cols = int(np.ceil(np.sqrt(n_features)))
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 5), sharey=True)
    axes = axes.flatten()

    median_values = X_train.median().values

    for i, feature in enumerate(feature_names):
        feature_data = X_train[feature].values
        min_val, max_val = feature_data.min(), feature_data.max()
        feature_grid = np.linspace(min_val, max_val, 100)

        X_grid = np.tile(median_values, (100, 1))
        X_grid[:, i] = feature_grid

        try:
            pdep, confi = m.partial_dependence(term=i, X=X_grid, width=0.95)
        except Exception as e:
            print(f"Error computing partial dependence for feature '{feature}': {e}")
            continue

        probs = expit(pdep)  # Convert log-odds to probabilities

        ax = axes[i]
        ax.plot(feature_grid, probs, label="Probability")
        ax.fill_between(feature_grid, expit(confi[:, 0]), expit(confi[:, 1]), alpha=0.2, label="95% CI")
        ax.set_title(f"Partial Dependence of {feature}")
        ax.set_ylabel("Probability")
        ax.set_ylim([0, 1])
        ax.grid(True)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(pad=5)
    plt.show()

if __name__ == "__main__":
    # Set target to 'hole' for binary prediction
    target, ws = "hole", 10
    data_file = f"data/{target}{ws}"
    big_df = load_prepared(data_file)

    # Split data into training and test sets
    train_df, test_df = train_test_split(big_df, test_size=0.2)

    # Define features and target
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # Check target distribution
    print(y_train.value_counts())

    # Train or load the model
    m = get_model(ws=ws, X_train=X_train, y_train=y_train)

    # Make predictions
    y_pred = m.predict(X_test)

    # Evaluate with classification metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Optionally, visualize partial dependence plots
    draw_partial_deps(m, X_train)