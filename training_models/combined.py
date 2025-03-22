import numpy as np
import pandas as pd
from pygam import LogisticGAM, GammaGAM
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.utils import compute_sample_weight


class CombinedGAM:
    def __init__(self, gam_binary_params=None, gam_positive_params=None):
        # Default parameters for binary classification GAM
        self.gam_binary_params = gam_binary_params or {
            'n_splines': 10,
            'lam': 0.6,
            'max_iter': 100
        }

        # Default parameters for positive values GAM
        self.gam_positive_params = gam_positive_params or {
            'n_splines': 15,
            'lam': 0.6,
            'max_iter': 100,
        }

        self.binary_gam = None
        self.positive_gam = None

    def _calculate_sample_weights(self, y):
        """Calculate sample weights to handle class imbalance."""
        # For binary classification: inverse of class frequency
        y_binary = (y > 0).astype(int)
        class_counts = np.bincount(y_binary)
        binary_weights = 1.0 / class_counts[y_binary]

        # For Gamma regression: no weighting (or use custom logic if needed)
        positive_weights = np.ones_like(y[y > 0])

        return binary_weights, positive_weights

    def fit(self, X, y):
        """Train combined model on data with zero-inflated target."""
        # Calculate sample weights
        binary_weights, positive_weights = self._calculate_sample_weights(y)

        # Train binary model for zero vs non-zero
        y_binary = (y > 0).astype(int)
        self.binary_gam = LogisticGAM(**self.gam_binary_params)
        self.binary_gam.fit(X, y_binary, weights=binary_weights)

        # Train Gamma model on positive values
        positive_mask = y > 0
        if positive_mask.sum() == 0:
            raise ValueError("No positive samples in training data")

        X_positive = X[positive_mask]
        y_positive = y[positive_mask]

        self.positive_gam = GammaGAM(**self.gam_positive_params)
        self.positive_gam.fit(X_positive, y_positive, weights=positive_weights)

        return self

    def predict(self, X):
        """Predict combined zero-inflated Gamma values."""
        if self.binary_gam is None or self.positive_gam is None:
            raise RuntimeError("Model must be trained before prediction")

        # Get probability of being non-zero
        prob_non_zero = self.binary_gam.predict_proba(X)

        # Get positive value predictions
        positive_pred = self.positive_gam.predict(X)

        # Combine predictions
        return prob_non_zero * positive_pred

    def evaluate(self, X, y_true):
        weights = compute_sample_weight('balanced', y_true)
        """Evaluate model performance with multiple metrics."""
        y_pred = self.predict(X)
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=weights)),
            'mae': mean_absolute_error(y_true, y_pred, sample_weight=weights),
            'explained_variance': 1 - (np.var(y_true - y_pred) / np.var(y_true))
        }

    def plot_components(self, X):
        """Visualize partial dependence plots for both models."""
        plt.figure(figsize=(15, 6))

        # Plot binary model components
        plt.subplot(1, 2, 1)
        for i, term in enumerate(self.binary_gam.terms):
            if term.isintercept:
                continue
            XX = self.binary_gam.generate_X_grid(term=i)
            pdep, confi = self.binary_gam.partial_dependence(term=i, X=XX, width=0.95)
            plt.plot(XX[:, term.feature], pdep, label=f'Feature {i + 1}')
        plt.title('Binary Model (Probability of Pothole)')
        plt.legend()

        # Plot Gamma model components
        plt.subplot(1, 2, 2)
        for i, term in enumerate(self.positive_gam.terms):
            if term.isintercept:
                continue
            XX = self.positive_gam.generate_X_grid(term=i)
            pdep, confi = self.positive_gam.partial_dependence(term=i, X=XX, width=0.95)
            plt.plot(XX[:, term.feature], pdep, label=f'Feature {i + 1}')
        plt.title('Gamma Model (Pothole Severity)')
        plt.legend()

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    from exploration.data_read import load_prepared
    from helpers import train_split_by_column

    # Load and prepare data
    target, window_size = "class", 10
    df = load_prepared(f"data/{target}{window_size}", sample_frac=0.5)
    X_train, y_train, X_test, y_test = train_split_by_column(df, target, 0.2)

    # Initialize and train combined model
    combined_model = CombinedGAM()
    combined_model.fit(X_train, y_train)

    # Evaluate
    train_metrics = combined_model.evaluate(X_train, y_train)
    test_metrics = combined_model.evaluate(X_test, y_test)

    print("Train Metrics:")
    print(
        f"RMSE: {train_metrics['rmse']:.2f}, MAE: {train_metrics['mae']:.2f}, Explained Variance: {train_metrics['explained_variance']:.2f}")
    print("\nTest Metrics:")
    print(
        f"RMSE: {test_metrics['rmse']:.2f}, MAE: {test_metrics['mae']:.2f}, Explained Variance: {test_metrics['explained_variance']:.2f}")

    # Visualize model components
    # combined_model.plot_components(X_train)