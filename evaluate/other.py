import glob
import os

import numpy as np
from tqdm import tqdm

from evaluate.ClassificationEval import evaluate_classification, X, y_test


def manage_models(models_path='models/*.pkl'):
    model_files = glob.glob(models_path)
    thresholds = np.arange(0.3, 0.7, 0.04)
    results = {}

    for model_path in tqdm(model_files):
        # print(type(model_path))

        model_name = os.path.basename(model_path).split('\\')[0]
        results[model_name] = []

        for i in thresholds:
            rmse, _, _ = evaluate_classification(X, y_test, model_path, i)
            results[model_name].append(rmse)

    mins = {n: np.min(score) for n, score in results.items()}
    # Rank models by their minimum RMSE (best to worst)
    sorted_mins = sorted(mins.items(), key=lambda x: x[1])

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
