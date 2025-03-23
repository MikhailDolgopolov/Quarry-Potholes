import glob
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import compute_sample_weight
from tqdm import tqdm

def evaluate_classification(X, y_test, filename) -> tuple[pd.DataFrame, LinearRegression, float]:
    # Load the model
    with open(filename, "rb") as f:
        model = pickle.load(f)

    # Make predictions
    y_pred = model.predict(X)

    # Create a dataframe with true and predicted values
    results_df = pd.DataFrame({
        'true_class': y_test,
        'prediction': y_pred
    })

    # Compute RMSE with balanced sample weights
    weights = compute_sample_weight('balanced', y_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=weights))

    # Fit linear regression on all true vs. predicted points
    reg = LinearRegression()
    reg.fit(y_test.values.reshape(-1, 1), y_pred, sample_weight=weights)

    return results_df, reg, rmse


def manage_models( x_test, y_test, models_path='models/*.pkl'):
    model_files = glob.glob(models_path)
    results = {}

    for model_path in tqdm(model_files):
        # print(type(model_path))

        model_name = os.path.basename(model_path).split('\\')[0]

        _, _, rmse = evaluate_classification(x_test, y_test, model_path)
        results[model_name] = rmse

    # Rank models by their minimum RMSE (best to worst)
    sorted_mins = sorted(results.items(), key=lambda x: x[1])

    # Print model rankings
    print("\nModel Rankings (Best to Worst):")
    print("Rank | Model Name".ljust(40) + " | Min RMSE")
    print("-" * 55)
    for rank, (model_name, min_score) in enumerate(sorted_mins, 1):
        print(f"{rank:4} | {model_name[:35]:35} | {min_score:.4f}")

    # Get user input for deletion
    try:
        num_to_delete = int(input("\nEnter number of worst models to delete (0 to cancel): "))
        if num_to_delete <= 0:
            return
    except ValueError:
        print("Invalid input. No models deleted.")
        return

    # Get worst performers to delete
    to_delete = sorted_mins[-num_to_delete:]

    # Confirm deletion
    print("\nWARNING: These models will be permanently deleted:")
    for model_name, score in to_delete:
        print(f"- {model_name} (RMSE: {score:.4f})")

    confirm = input("\nConfirm deletion? (y/n): ").lower()
    if confirm != 'y':
        print("Deletion canceled.")
        return

    # Delete files
    deleted_count = 0
    for model_name, _ in to_delete:
        # Find matching model file
        for model_path in model_files:
            if os.path.basename(model_path) == model_name:
                try:
                    os.remove(model_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {model_name}: {str(e)}")
                break

    print(f"\nSuccessfully deleted {deleted_count}/{num_to_delete} models")
