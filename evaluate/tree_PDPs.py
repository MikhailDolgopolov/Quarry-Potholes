import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from exploration.data_read import load_preprocessed

# Load data
target = 'severity'  # Consider renaming if it's a continuous target
df = load_preprocessed(f'data/{target}10', sample_frac=0.3)
X, y = df.drop(columns=[target]), df[target]  # Renamed y_test to y

# Load model
model_path = "HGBR_[l2_regularization0.5][learning_rate0.6][max_iter200][min_samples_leaf5][random_state42][scoringneg_mean_absolute_error][tol0.01]_top1_21.pkl"
with open(f'models/{model_path}', 'rb') as f:
    model = pickle.load(f)


# Dynamic figure size
n_features = len(X.columns)
n_cols = 4
n_rows = round(np.ceil(n_features/n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
# Create partial dependence plots for all features
display = PartialDependenceDisplay.from_estimator(
    model,
    X.sample(1000),
    features=np.arange(n_features),
    feature_names=X.columns,  # Added feature names
    method='brute',
    grid_resolution=50,  # Increased for smoother plots
    n_jobs=-1,
    line_kw={'linewidth': 2},
    kind='average',  # Simpler plot; or use 'both' with ice_kw={'alpha': 0.2'}
    ax=axes.ravel()[:n_features]
)

# Adjust layout and title
plt.subplots_adjust(hspace=0.2, wspace=0.3)

# Save with dynamic name
output_name = f'images/partial_dependence_1.png'
plt.savefig(output_name, bbox_inches='tight', dpi=300)
# plt.show()
plt.close()