import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from clustering.LVQ import predict_with_LVQ
from exploration.data_read import load_prepared
from helpers import train_split_by_column

# Load dataset
target, ws = 'class', 5
df = load_prepared(f'data/{target}{ws}')
X, y = df.drop(columns=target), df[target]
y_binary = np.where(y > 0, 1, 0)

# Iterate through all LVQ models
for model_path in glob.glob(f'models/LVQs/*hole*.pkl'):
    # print(f'\nEvaluating {model_path}')
    hole_pred = predict_with_LVQ(model_path, X)
    print(classification_report(y_binary, hole_pred))
    accuracy = classification_report(y_binary, hole_pred, output_dict=True)['accuracy']

    predicted_hole_df = df[hole_pred > 0]  # Positive predictions
    predicted_non_hole_df = df[hole_pred == 0]  # Negative predictions

    # Compute overall class distributions
    overall_counts = y.value_counts()
    hole_counts = predicted_hole_df[target].value_counts()
    non_hole_counts = predicted_non_hole_df[target].value_counts()

    # Compute proportions
    hole_proportion = hole_counts.divide(overall_counts)
    non_hole_proportion = non_hole_counts.divide(overall_counts)

    # Setup side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    title = model_path.split('LVQs/')[-1].split('.')[0]
    title += f'\nAccuracy: {accuracy:.3f}'
    fig.suptitle(title, fontsize=14)

    # Plot positive (hole) predictions
    axes[0].bar(hole_proportion.index, hole_proportion, color='red', alpha=0.7, width=20)
    axes[0].set_title("Proportion in Predicted Positive Cases")
    axes[0].set_xticks(hole_proportion.index)
    axes[0].set_ylim(0, 1)

    # Plot negative (non-hole) predictions
    axes[1].bar(non_hole_proportion.index, non_hole_proportion, color='blue', alpha=0.7, width=20)
    axes[1].set_title("Proportion in Predicted Negative Cases")
    axes[1].set_xticks(non_hole_proportion.index)
    axes[1].set_ylim(0, 1)

    # Show plots
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    plt.show()
