from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from sklvq import GLVQ

def lvq_class_separation(model: GLVQ, feature_names, accuracy=None, save_path: str | Path = None):
    feature_names = np.array(feature_names)
    # Get unique classes and their prototype counts
    unique_classes = np.unique(model.prototypes_labels_)
    class_counts = {cls: np.sum(model.prototypes_labels_ == cls) for cls in unique_classes}
    counts_str = ", ".join([f"{cls}: {class_counts[cls]}" for cls in sorted(unique_classes)])
    prototype_desc = f"({counts_str})"

    # Compute mean prototypes for each class
    mean_prototypes = {cls: np.mean(model.prototypes_[model.prototypes_labels_ == cls], axis=0)
                       for cls in unique_classes}
    # Compute feature importance (range between class means)
    feature_importance = pd.Series(
        [np.ptp([mean_prototypes[cls][j] for cls in unique_classes]) for j in range(len(feature_names))],
        index=feature_names
    ).sort_values(ascending=False)
    top_features = feature_importance.index

    # Color mapping for classes
    n_classes = len(unique_classes)
    cmap = plt.get_cmap('Set1')
    colors = {cls: cmap(i / n_classes) for i, cls in enumerate(sorted(unique_classes))}

    # Setup subplots
    n_cols = 3
    n_rows = int(np.ceil(len(top_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle(f"Prototype Values by Feature {prototype_desc}", fontsize=16, fontweight="bold")
    if accuracy is not None:
        fig.text(0.5, 0.93, f"Overall Accuracy: {accuracy:.2f}", fontsize=12, ha="center", style="italic")
    axes = axes.flatten()

    # Plot prototype values for each feature
    for i, feature in enumerate(top_features):
        ax = axes[i]
        feat_idx = np.where(feature_names == feature)[0][0]

        for j, cls in enumerate(sorted(unique_classes)):
            prototypes = model.prototypes_[model.prototypes_labels_ == cls][:, feat_idx]
            if len(prototypes) > 1:
                # Multiple prototypes: plot mean with error bars
                mean_val = np.mean(prototypes)
                std_val = np.std(prototypes)
                ax.errorbar(j, mean_val, yerr=std_val, fmt='o', color=colors[cls], capsize=5, label=f'Class {cls}')
            else:
                # Single prototype: plot as a point
                ax.plot(j, prototypes[0], 'o', color=colors[cls], label=f'Class {cls}')

        ax.set_xticks(range(len(unique_classes)))
        ax.set_xticklabels([str(cls) for cls in sorted(unique_classes)])
        ax.set_title(f"{feature}")
        ax.grid(True, axis='y', linestyle="--", alpha=0.6)

    # Add legend
    legend_patches = [Patch(color=colors[cls], label=f"Class {cls}") for cls in sorted(unique_classes)]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=10)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.90], h_pad=0.4, w_pad=0.2)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


def lvq_radar_chart(model: GLVQ, feature_names, accuracy=None, save_path=None, top_n_features=0):
    """
    Create a radar chart to visualize GLVQ prototypes.

    Parameters:
    - model: Trained GLVQ model from sklvq
    - feature_names: List or array of feature names
    - accuracy: Optional model accuracy to display
    - save_path: Optional path to save the plot
    - top_n_features: Number of top features to display (default: 5)
    """
    feature_names = np.array(feature_names)
    unique_classes = np.unique(model.prototypes_labels_)

    # Count prototypes per class for the title
    class_counts = {cls: np.sum(model.prototypes_labels_ == cls) for cls in unique_classes}
    counts_str = ", ".join([f"Class {cls}: {class_counts[cls]}" for cls in sorted(unique_classes)])
    title = f"Radar Chart of GLVQ Prototypes\n({counts_str})"

    # Compute mean prototypes for each class
    mean_prototypes = {
        cls: np.mean(model.prototypes_[model.prototypes_labels_ == cls], axis=0)
        for cls in unique_classes
    }

    # Find the most important features (based on range of prototype values)
    feature_importance = pd.Series(
        [np.ptp([mean_prototypes[cls][j] for cls in unique_classes]) for j in range(len(feature_names))],
        index=feature_names
    ).sort_values(ascending=False)
    if top_n_features == 0:
        top_n_features = len(feature_names)
    top_features = feature_importance.index[:top_n_features].tolist()
    selected_indices = [np.where(feature_names == feat)[0][0] for feat in top_features]

    # Normalize prototype values for the selected features
    all_values = np.vstack([mean_prototypes[cls][selected_indices] for cls in unique_classes])
    min_val, max_val = all_values.min(), all_values.max()
    normalized_prototypes = {
        cls: (mean_prototypes[cls][selected_indices] - min_val) / (max_val - min_val)
        for cls in unique_classes
    }

    # Set up the radar chart
    n_features = len(top_features)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    if accuracy is not None:
        fig.text(0.5, 0.95, f"Accuracy: {accuracy:.2f}", ha="center", fontsize=10)

    # Plot each class
    for cls in sorted(unique_classes):
        values = normalized_prototypes[cls].tolist()
        values += values[:1]  # Close the polygon
        ax.plot(angles, values, linewidth=2, label=f"Class {cls}")
        ax.fill(angles, values, alpha=0.2)  # Light fill for visibility

    # Add feature labels and legend
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_features, fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    # Adjust layout and display or save
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)