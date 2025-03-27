import os
import pickle
from pprint import pprint

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklvq import GLVQ

from exploration.data_read import load_prepared
from sklearn.metrics import mean_absolute_error, classification_report

from helpers import train_split_by_column

def importances(x, y, switch, ps)->pd.Series:
    if switch:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(x)
    print("Training GLVQ model...")
    model = GLVQ(prototype_n_per_class=np.array(ps))
    model.fit(x, y)

    prototype_labels = model.prototypes_labels_
    prototypes_even = prototypes_original[prototype_labels == 0]
    prototypes_hole = prototypes_original[prototype_labels == 1]

    # Compute mean prototypes.
    mean_prototype_even = np.mean(prototypes_even, axis=0)
    mean_prototype_hole = np.mean(prototypes_hole, axis=0)

    # === Feature Importance Analysis ===
    feature_names = X_train_oversampled.columns
    differences = np.abs(mean_prototype_hole - mean_prototype_even)
    feature_importance = pd.Series(differences, index=feature_names).sort_values(ascending=False)
    return feature_importance

if __name__ == '__main__':
    # === Configuration ===
    prototypes = [2, 4]
    prots = f'[{",".join(map(str, prototypes))}]'
    target, ws = 'hole', 10
    # Switch to enable or disable scaling
    use_scaler = False

    model_dir = 'models/LVQs'
    scaled = 'scaled' if use_scaler else 'original'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'glvq_{target}{ws}-{scaled}-{prots}.joblib')
    scaler_path = os.path.join(model_dir, f'scaler_{target}{ws}-{scaled}-{prots}.joblib')


    # You can also add a switch for using a pre-trained model if needed.
    use_pretrained_model = True

    # === Data Loading & Splitting ===
    df = load_prepared(f'data/{target}{ws}', keep_latlon=False, sample_frac=0.5)
    X_train, y_train, X_test, y_test = train_split_by_column(df, target, 0.5)

    # === Handling Class Imbalance by Oversampling Minority ===
    minority_class = 1
    majority_class = 0
    X_minority = X_train[y_train == minority_class]
    y_minority = y_train[y_train == minority_class]
    X_majority = X_train[y_train == majority_class]
    y_majority = y_train[y_train == majority_class]

    n_majority = len(X_majority)
    X_minority_oversampled, y_minority_oversampled = resample(
        X_minority, y_minority,
        replace=True,
        n_samples=int(n_majority * 1.2),
        random_state=42
    )

    X_train_oversampled = pd.concat([X_majority, X_minority_oversampled]).reset_index(drop=True)
    y_train_oversampled = pd.concat([y_majority, y_minority_oversampled]).reset_index(drop=True)

    # === Scaling (Optional) ===
    if use_scaler:
        # If using a scaler, try to load a pre-trained model and scaler if available.
        if use_pretrained_model and os.path.exists(model_path) and os.path.exists(scaler_path):
            print(f"Loading pre-trained model and scaler from {model_path} and {scaler_path}...")
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            X_train_scaled = scaler.transform(X_train_oversampled)
            X_test_scaled = scaler.transform(X_test)
        else:
            # Fit the scaler on training data and transform both train and test sets.
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_oversampled)
            X_test_scaled = scaler.transform(X_test)
    else:
        # If not scaling, use the original data
        X_train_scaled = X_train_oversampled.values
        X_test_scaled = X_test.values

    # === Training the GLVQ Model ===
    if use_pretrained_model and os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}...")
        model = joblib.load(model_path)
    else:
        print("Training GLVQ model...")
        model = GLVQ(prototype_n_per_class=np.array(prototypes))
        model.fit(X_train_scaled, y_train_oversampled.to_numpy())
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        # Save the scaler only if used.
        if use_scaler:
            joblib.dump(scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")

    # === Predict & Evaluate ===
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

    # === Mapping Prototypes Back to Original Space (if scaler was used) ===
    if use_scaler:
        prototypes_scaled = model.prototypes_
        prototypes_original = scaler.inverse_transform(prototypes_scaled)
    else:
        prototypes_original = model.prototypes_

    # Assign prototypes to classes.
    prototype_labels = model.prototypes_labels_
    prototypes_even = prototypes_original[prototype_labels == 0]
    prototypes_hole = prototypes_original[prototype_labels == 1]

    # Compute mean prototypes.
    mean_prototype_even = np.mean(prototypes_even, axis=0)
    mean_prototype_hole = np.mean(prototypes_hole, axis=0)

    # === Feature Importance Analysis ===
    feature_names = X_train_oversampled.columns
    differences = np.abs(mean_prototype_hole - mean_prototype_even)
    feature_importance = pd.Series(differences, index=feature_names).sort_values(ascending=False)
    top_features = feature_importance.index[:8]

    # If scaling was applied, we need to invert the scaling for visualization.
    if use_scaler:
        X_train_original = scaler.inverse_transform(X_train_scaled)
    else:
        X_train_original = X_train_scaled
    X_train_df = pd.DataFrame(X_train_original, columns=feature_names)
    X_train_df[target] = y_train_oversampled.values

    ranges_even = X_train_df[X_train_df[target] == 0].agg(['mean', 'std']).drop(columns=target)
    ranges_hole = X_train_df[X_train_df[target] == 1].agg(['mean', 'std']).drop(columns=target)

    ranges_even_df = pd.DataFrame({
        'lower': ranges_even.loc['mean'] - ranges_even.loc['std'],
        'upper': ranges_even.loc['mean'] + ranges_even.loc['std']
    }, index=feature_names)

    ranges_hole_df = pd.DataFrame({
        'lower': ranges_hole.loc['mean'] - ranges_hole.loc['std'],
        'upper': ranges_hole.loc['mean'] + ranges_hole.loc['std']
    }, index=feature_names)

    print("Top Significant Features (based on mean prototype differences):")
    print(feature_importance[top_features])

    # === Visualization ===
    n_cols = 3  # Number of columns for subplots
    n_rows = int(np.ceil(len(top_features) / n_cols))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 5 * n_rows))
    fig.suptitle(f"Significant Features and Their Ranges ({prots} Prototypes)", fontsize=14)
    ax_flat = ax.flatten()

    bar_params = {'alpha': 0.3, 'width': 0.25}
    for i, feature in enumerate(top_features):
        # Plot each prototype for even class.
        for proto in prototypes_even:
            ax_flat[i].bar('even', proto[feature_names == feature], color='blue', **bar_params)
        # Plot each prototype for hole class.
        for proto in prototypes_hole:
            ax_flat[i].bar('hole', proto[feature_names == feature], color='red', **bar_params)

        # Plot mean prototypes.
        ax_flat[i].bar('even', mean_prototype_even[feature_names == feature], color='blue', label='even mean',
                       **bar_params)
        ax_flat[i].bar('hole', mean_prototype_hole[feature_names == feature], color='red', label='hole mean',
                       **bar_params)

        ax_flat[i].set_title(f"{feature}")
        ax_flat[i].set_ylabel("Feature Value")
        ax_flat[i].legend()

    # Hide any unused subplots.
    for j in range(i + 1, len(ax_flat)):
        ax_flat[j].set_visible(False)

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()
