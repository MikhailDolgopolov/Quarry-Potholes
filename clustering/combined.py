import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from clustering.LVQ import predict_with_LVQ
from exploration.data_read import load_prepared
from helpers import train_split_by_column

# Load dataset
target, ws = 'class', 10
df = load_prepared(f'data/{target}{ws}')
X, y = df.drop(columns=target), df[target]
y_binary = np.where(y > 0, 1, 0)
overall_counts = y.value_counts().sort_index()
classes = overall_counts.index
bar_w=20
for model_path in glob.glob(f'models/LVQs/*hole{ws}*.pkl'):
    # Make predictions and evaluate
    hole_pred = predict_with_LVQ(model_path, X)
    # print(classification_report(y_binary, hole_pred))
    report = classification_report(y_binary, hole_pred, output_dict=True)
    accuracy = report['accuracy']
    f1_pothole = report['1']['f1-score']
    f1_nonhole = report['0']['f1-score']

    # Compute confusion matrix counts
    TN_count = ((y_binary == 0) & (hole_pred == 0)).sum()
    FN_count = ((y_binary == 1) & (hole_pred == 0)).sum()
    FP_count = ((y_binary == 0) & (hole_pred == 1)).sum()
    TP_count = ((y_binary == 1) & (hole_pred == 1)).sum()

    # Compute proportions for predicted negative cases
    total_pred_neg = TN_count + FN_count
    TN_proportion = TN_count / total_pred_neg if total_pred_neg > 0 else 0
    FN_proportion = FN_count / total_pred_neg if total_pred_neg > 0 else 0

    # Compute proportions for predicted positive cases
    total_pred_pos = FP_count + TP_count
    FP_proportion = FP_count / total_pred_pos if total_pred_pos > 0 else 0
    TP_proportion = TP_count / total_pred_pos if total_pred_pos > 0 else 0

    # Compute proportions for original classes in predictions
    predicted_hole_df = df[hole_pred > 0]
    predicted_non_hole_df = df[hole_pred == 0]
    hole_counts = predicted_hole_df[target].value_counts()
    non_hole_counts = predicted_non_hole_df[target].value_counts()
    hole_proportion = (hole_counts.reindex(classes, fill_value=0) / overall_counts).fillna(0)
    non_hole_proportion = (non_hole_counts.reindex(classes, fill_value=0) / overall_counts).fillna(0)

    # Setup 2x2 plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    title = f'Using window {ws} data to test\n'
    title += model_path.split('LVQs/')[-1].split('.')[0]
    title += f'\nAccuracy: {accuracy:.3f} | F1-Hole: {f1_pothole:.3f} | F1-Non-Hole: {f1_nonhole:.3f}'
    fig.suptitle(title, fontsize=14)

    # Top-Left: Composition of Predicted Negative Cases
    neg_bars = ['TN', 'FN']
    neg_proportions = [TN_proportion, FN_proportion]
    neg_colors = ['blue', 'red']
    bars0 = axes[0, 0].bar(neg_bars, neg_proportions, color=neg_colors, alpha=0.7)
    axes[0, 0].set_title("Composition of Predicted Negative Cases")
    axes[0, 0].set_ylim(0, 1)
    for bar in bars0:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom')

    # Top-Right: Composition of Predicted Positive Cases
    pos_bars = ['FP', 'TP']
    pos_proportions = [FP_proportion, TP_proportion]
    pos_colors = ['red', 'blue']
    bars1 = axes[0, 1].bar(pos_bars, pos_proportions, color=pos_colors, alpha=0.7)
    axes[0, 1].set_title("Composition of Predicted Positive Cases")
    axes[0, 1].set_ylim(0, 1)
    for bar in bars1:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom')

    # Bottom-Left: Proportion of Each Class in Predicted Negative Cases
    axes[1, 0].bar(classes, non_hole_proportion[classes], alpha=0.7, width=bar_w)
    axes[1, 0].set_title("Proportion of Each Class in Predicted Negative Cases")
    axes[1, 0].set_xticks(classes)
    axes[1, 0].set_ylim(0, 1)

    # Bottom-Right: Proportion of Each Class in Predicted Positive Cases
    axes[1, 1].bar(classes, hole_proportion[classes], alpha=0.7, width=bar_w)
    axes[1, 1].set_title("Proportion of Each Class in Predicted Positive Cases")
    axes[1, 1].set_xticks(classes)
    axes[1, 1].set_ylim(0, 1)

    # Adjust layout and display
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()