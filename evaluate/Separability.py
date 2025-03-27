import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from exploration.data_read import load_prepared
from helpers import train_split_by_column

target, ws = 'hole', 5
# Load data and split (replace these with your actual functions)
df = load_prepared(f'data/{target}{ws}', keep_latlon=False, sample_frac=1)
X_train, y_train, X_test, y_test = train_split_by_column(df, target, 0.5)

# ----- 1. Fisher Score Calculation -----
def fisher_score(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Compute the Fisher score for each feature.
    Higher Fisher score indicates better class separability.
    """
    classes = np.unique(y)
    scores = {}
    for col in X.columns:
        overall_mean = X[col].mean()
        numerator = 0.0
        denominator = 0.0
        for cls in classes:
            X_cls = X.loc[y == cls, col]
            n_cls = len(X_cls)
            mean_cls = X_cls.mean()
            var_cls = X_cls.var()
            numerator += n_cls * (mean_cls - overall_mean) ** 2
            denominator += n_cls * var_cls
        # Avoid division by zero by adding a small constant.
        scores[col] = numerator / (denominator + 1e-8)
    return scores
#
# fisher_scores = fisher_score(X_train, y_train)
# print("Fisher Scores (higher is better):")
# print(pd.Series(fisher_scores).sort_values(ascending=False))


# ----- 2. Bhattacharyya Distance Calculation -----
def bhattacharyya_distance(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Compute the Bhattacharyya distance for each feature (binary classification only).
    Lower values indicate less separability (more overlap).
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("Bhattacharyya distance is defined for binary classes only.")
    cls1, cls2 = classes
    distances = {}
    for col in X.columns:
        X1 = X.loc[y == cls1, col]
        X2 = X.loc[y == cls2, col]
        mu1, mu2 = X1.mean(), X2.mean()
        sigma1, sigma2 = X1.std(), X2.std()
        # Prevent division by zero by ensuring a minimum standard deviation.
        sigma1 = sigma1 if sigma1 > 0 else 1e-8
        sigma2 = sigma2 if sigma2 > 0 else 1e-8
        term1 = 0.25 * np.log(0.25 * (((sigma1 / sigma2) ** 2) + ((sigma2 / sigma1) ** 2) + 2))
        term2 = 0.25 * ((mu1 - mu2) ** 2) / (sigma1 ** 2 + sigma2 ** 2)
        distances[col] = term1 + term2
    return distances

# bhatt_distances = bhattacharyya_distance(X_train, y_train)
# print("\nBhattacharyya Distances (lower is better):")
# print(pd.Series(bhatt_distances).sort_values(ascending=True))


def jeffries_matusita_distance(X: pd.DataFrame, y: pd.Series) -> float:
    """
    Compute the Jeffries-Matusita distance between two classes using a multivariate
    Gaussian assumption. The JM distance is defined as:

        JM = sqrt(2 * (1 - exp(-B)))

    where B is the Bhattacharyya distance given by:

        B = (1/8) * (mu2 - mu1)^T * inv((cov1+cov2)/2) * (mu2 - mu1) +
            (1/2) * log( det((cov1+cov2)/2) / sqrt(det(cov1)*det(cov2)) )

    Parameters:
    -----------
    X : pd.DataFrame
        DataFrame of features.
    y : pd.Series
        Binary class labels.

    Returns:
    --------
    float
        The Jeffries-Matusita distance (bounded between 0 and sqrt(2)).
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("Jeffries-Matusita distance is defined for binary classification only.")

    # Separate data for the two classes
    X1 = X.loc[y == classes[0]]
    X2 = X.loc[y == classes[1]]

    # Compute mean vectors for each class
    mu1 = X1.mean().to_numpy()
    mu2 = X2.mean().to_numpy()

    # Compute covariance matrices for each class
    cov1 = np.cov(X1.T)
    cov2 = np.cov(X2.T)

    # Average covariance matrix
    cov_mean = (cov1 + cov2) / 2.0

    # Compute the first term: Mahalanobis distance between the means
    diff = mu2 - mu1
    term1 = 1.0 / 8.0 * np.dot(diff.T, np.linalg.solve(cov_mean, diff))

    # Prevent issues with log(0) by adding a small epsilon
    eps = 1e-8
    det_cov_mean = np.linalg.det(cov_mean) + eps
    det_cov1 = np.linalg.det(cov1) + eps
    det_cov2 = np.linalg.det(cov2) + eps

    term2 = 0.5 * np.log(det_cov_mean / np.sqrt(det_cov1 * det_cov2))

    # Bhattacharyya distance
    B = term1 + term2

    # Jeffries-Matusita distance: bounded between 0 and sqrt(2)
    JM = np.sqrt(2 * (1 - np.exp(-B)))
    return JM

# ----- 3. Cross-Validated Logistic Regression Accuracy -----
# This is an empirical proxy: if the best simple model can't separate the classes well,
# then the intrinsic separability might be low.
lr = LogisticRegression(max_iter=1000)
cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='f1_weighted')
print("\nLogistic Regression CV Accuracy: {:.3f} ± {:.3f}".format(np.mean(cv_scores), np.std(cv_scores)))

jm_distance = jeffries_matusita_distance(X_train, y_train)
print("Jeffries-Matusita Distance:", jm_distance)


def multivariate_bhattacharyya_distance(X: pd.DataFrame, y: pd.Series) -> float:
    """
    Compute the multivariate Bhattacharyya distance between two classes
    in a dataset. Assumes the features for each class can be modeled as
    multivariate Gaussian distributions.

    Parameters:
    -----------
    X : pd.DataFrame
        DataFrame of features.
    y : pd.Series
        Series containing binary class labels.

    Returns:
    --------
    float
        The Bhattacharyya distance between the two classes.
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("This function is defined for binary classification only.")

    # Separate the data for the two classes
    X1 = X.loc[y == classes[0]]
    X2 = X.loc[y == classes[1]]

    # Compute mean vectors
    mu1 = X1.mean().to_numpy()
    mu2 = X2.mean().to_numpy()

    # Compute covariance matrices
    cov1 = np.cov(X1.T)
    cov2 = np.cov(X2.T)

    # Average covariance matrix
    cov_mean = (cov1 + cov2) / 2.0

    # Compute the first term of the Bhattacharyya distance
    diff = mu2 - mu1
    term1 = 1.0 / 8.0 * np.dot(diff.T, np.linalg.solve(cov_mean, diff))

    # Compute the second term; add a small constant to determinants to prevent log(0)
    eps = 1e-8
    det_cov_mean = np.linalg.det(cov_mean) + eps
    det_cov1 = np.linalg.det(cov1) + eps
    det_cov2 = np.linalg.det(cov2) + eps
    term2 = 0.5 * np.log(det_cov_mean / np.sqrt(det_cov1 * det_cov2))

    return term1 + term2

# # Compute the overall separability metric
# overall_separability = multivariate_bhattacharyya_distance(X_train, y_train)
# print("Multivariate Bhattacharyya Distance:", overall_separability)