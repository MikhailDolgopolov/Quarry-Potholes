import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
from sklearn.utils import compute_sample_weight
import pandas as pd
import pickle
from joblib import Parallel, delayed
from pprint import pprint

from xgboost import XGBRegressor, XGBClassifier

from exploration.data_read import load_preprocessed

cols=["acc_X_std",
            "acc_X_var",
            "acc_Z_iqr",
            "acc_X_iqr",
            "acc_Y_std",
            "acc_Y_kurt",
            "acc_Z_std",
            "acc_Y_std",
            "acc_X_kurt",
            "acc_Y_std"]


def train_evaluate_xgb(params, X_t, y_t, X_e, y_e):
    """
    Train an XGBRegressor with given hyperparameters, and return evaluation metrics.
    """
    try:
        # Get default parameters for reference
        default_params = XGBRegressor().get_params()

        # Create model with our parameters, plus fixed values for n_estimators and random_state.
        model = XGBRegressor(
            **params,
            n_estimators=100,
            random_state=42
        )
        # Fit model (convert DataFrames/Series to numpy arrays)
        X_t_np = X_t.to_numpy() if isinstance(X_t, pd.DataFrame) else X_t
        y_t_np = y_t.to_numpy() if isinstance(y_t, pd.Series) else y_t
        model.fit(X_t_np, y_t_np)

        # Predict on evaluation set
        X_e_np = X_e.to_numpy() if isinstance(X_e, pd.DataFrame) else X_e
        test_pred = model.predict(X_e_np)
        test_rmse = np.sqrt(mean_squared_error(y_e, test_pred))
        test_mae = mean_absolute_error(y_e, test_pred)

        # Identify non-default parameters for reporting
        non_default_params = {
            k: v for k, v in params.items()
            if str(v) != str(default_params.get(k, None))
        }
        return {
            'params': non_default_params,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'model': model
        }
    except Exception as e:
        print(f"Error with params {params}: {str(e)}")
        return None


def xgb_grid_search(param_grid: dict, df: pd.DataFrame, target: str, test_frac=0.2, n_jobs=4,
                    save_num=3, print_num=3, name="XGBR"):
    """
    Custom grid search for XGBRegressor.
    """
    param_combinations = list(ParameterGrid(param_grid))
    print(f"Grid search combinations: {len(param_combinations)}")

    # Split data once upfront
    train_df, test_df = train_test_split(df, test_size=test_frac, random_state=42)
    X_train, y_train = train_df[cols], train_df[target]
    X_test, y_test = test_df[cols], test_df[target]

    # Optionally, compute sample weights (here, not used in XGBRegressor by default)
    train_weights = compute_sample_weight('balanced', y_train)
    test_weights = compute_sample_weight('balanced', y_test)

    # Run grid search in parallel
    search_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(train_evaluate_xgb)(params, X_train, y_train, X_test, y_test)
        for params in param_combinations
    )

    # Remove None results and sort by RMSE
    results = [res for res in search_results if res is not None]
    sorted_results = sorted(results, key=lambda x: x['test_rmse'])

    # Save top models
    for idx, result in enumerate(sorted_results[:save_num], 1):
        model = result['model']
        pstr = ''.join([f'[{k}={v}]' for k, v in result['params'].items()])
        model_name = f"{name}_{pstr}_top{idx}_{round(result['test_mae'], 2)}.pkl"
        with open(f"models/{model_name}", "wb") as f:
            pickle.dump(model, f)
        print(f"Saved: {model_name}")

    pprint(sorted_results[:print_num])
    return sorted_results


if __name__ == "__main__":
    pd.set_option('display.max_columns', 25)
    pd.set_option('display.width', 1000)

    target = 'severity'
    big_df = load_preprocessed(f"data/engineered/30peaks/rolled7", sample_frac=1)

    # Ensure your target and features are numeric as needed
    # big_df[cols] = big_df[cols].select_dtypes(include=[np.number])
    # big_df[target] = pd.to_numeric(big_df[target], errors='coerce')

    # You can try a separate simple XGB run to get baseline metrics:
    cols = big_df.columns.drop(['severity', 'pothole'])
    print(cols)
    X_train, X_test, y_train, y_test = train_test_split(big_df[cols], big_df[target], test_size=0.2, random_state=42)

    if target == 'pothole':
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        scale_pos_weight = neg / pos
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            scale_pos_weight=scale_pos_weight  # handle class imbalance
        )
        model.fit(X_train.to_numpy(), y_train.to_numpy())

        y_pred = model.predict(X_test.to_numpy())
        print(classification_report(y_test, y_pred))
    elif target == 'severity':
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
        )
        model.fit(X_train.to_numpy(), y_train.to_numpy())
        y_pred = model.predict(X_test.to_numpy())
        print(mean_absolute_error(y_test, y_pred))