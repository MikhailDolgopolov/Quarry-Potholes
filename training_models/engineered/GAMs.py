import os
import pickle
import numpy as np
import pandas as pd
from fontTools.misc.cython import returns
from matplotlib import pyplot as plt
from pygam import LinearGAM
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid, train_test_split
from joblib import Parallel, delayed
from sklearn.utils import compute_sample_weight

from exploration.data_read import load_engineered_data
from helpers import train_split_by_column


def train_gam(params: dict, X_t: pd.DataFrame, y_t: pd.Series,
              X_e: pd.DataFrame, y_e: pd.Series,
              train_weights=None, eval_weights=None) -> tuple[any, dict, float]:
    """Train a single GAM model and return it with its validation score."""
    if train_weights is None:
        train_weights = compute_sample_weight('balanced', y_t)
    if eval_weights is None:
        eval_weights = compute_sample_weight('balanced', y_e)
    try:
        model = LinearGAM(**params)
        model.fit(X_t, y_t, weights=train_weights)
        score = np.sqrt(mean_squared_error(y_e, model.predict(X_e), sample_weight=eval_weights))
        return model, params, score
    except Exception as e:
        print(f"Failed training with params {params}: {e}")
        return None, dict(), np.inf


def search_grid(X_t, y_t, X_e, y_e,
                      param_grid: dict, n_jobs: int = 4) -> tuple[LinearGAM, dict, float]:
    """Train or load the best LinearGAM model with parallelized hyperparameter tuning."""


    # Parallelized grid search
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(train_gam)(params, X_t, y_t, X_e, y_e) for params in ParameterGrid(param_grid)
    )

    # Find the best model
    best_model, best_param, best_score = None, dict(), np.inf
    for model, p, score in results:
        if score < best_score:
            best_model, best_param, best_score = model, p, score

    # Save the best model
    if best_model is not None:
        pstr = ''.join([f'[{k}{v}]' for k, v in best_param.items()])
        best_model_path = os.path.join('models', f"LGAM_{pstr}_{round(best_score)}.pkl")
        with open(best_model_path, "wb") as f:
            pickle.dump(best_model, f)
        print(f"Best model saved to {best_model_path} with RMSE: {best_score:.2f}")

    return best_model, best_param, best_score


def plot_partial_dependencies(model: LinearGAM, X: pd.DataFrame):
    """Visualize partial dependencies with confidence intervals."""
    n_features = X.shape[1]
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for i, (name, values) in enumerate(X.items()):
        ax = axes[i]
        XX = model.generate_X_grid(term=i)
        pdep, confi = model.partial_dependence(term=i, X=XX, width=0.95)

        ax.plot(XX[:, i], pdep, label="Effect")
        ax.fill_between(XX[:, i], confi[:, 0], confi[:, 1], alpha=0.2)
        ax.set_title(f"Partial dependence: {name}")
        ax.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.1)
    plt.show()


if __name__ == "__main__":
    # Data preparation
    target, window_size = "class", 10
    df = load_engineered_data(f"data/{target}{window_size}")
    X_train, y_train, X_test, y_test = train_split_by_column(df, target, 0.2)

    remove = [
        'acc_X_std', 'acc',
              'acc_Z_max', 'acc_Z', 'acc_Z_std',
              'fb_tilt_range', 'fb_tilt_var', 'tilt_var']
    X_train = X_train.drop(columns = remove)
    X_test = X_test.drop(columns = remove)
    # Hyperparameter grid for LinearGAM
    param_grid = {
        "n_splines": [5, 10, 15, 20, 30],  # Number of splines per feature
        "lam": [5.0, 10, 20, 30],  # Regularization strength
        # "max_iter": [50, 100, 200]  # Maximum iterations for optimization
    }

    # Model training/loading
    # print("Grid search combinations:", len(ParameterGrid(param_grid)))
    # model, param, score = search_grid(X_train, y_train, X_test, y_test, param_grid, n_jobs=4)
    # model, param, score = train_gam({
    #     'lam':0.6, 'n_splines': 30
    # }, X_train, y_train, X_test, y_test)
    #
    # print("RMSE:", score)
    # plot_partial_dependencies(model, X_train)
    # if score<38:
    #     pstr = ''.join([f'[{k}{v}]' for k, v in param.items()])
    #     best_model_path = os.path.join('models', f"LGAM_{pstr}_{round(score)}.pkl")
    #     with open(best_model_path, "wb") as f:
    #         pickle.dump(model, f)

    # filename = 'models/LGAM_[lam20][n_splines30]_37.pkl'
    # with open(filename, "rb") as f:
    #     model = pickle.load(f)
    # plot_partial_dependencies(model, X_train)

    tweedie_reg = TweedieRegressor(power=1.5, alpha=0.5)  # 1 < power < 2 for zero-inflated data
    tweedie_reg.fit(X_train, y_train, sample_weight=compute_sample_weight('balanced', y_train))

    y_pred = tweedie_reg.predict(X_test)
    score = np.sqrt(mean_squared_error(y_test, y_pred,
                                       sample_weight=compute_sample_weight('balanced', y_test)))
    print('Tweedie RMSE:', score )