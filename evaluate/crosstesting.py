import pickle

import numpy as np
from sklearn.utils import compute_sample_weight
from tqdm import tqdm

from data_read import load_prepared

if __name__ == '__main__':
    df = load_prepared(f'data/hole10', sample_frac=0.3)
    X, y_test = df.drop(columns=['hole']), df['hole']
    sample_weights = compute_sample_weight('balanced', y_test)

    filename='models/HGBR_[l2_regularization0.3][max_depthNone]_top3_21.pkl'
    with open(filename, "rb") as f:
        model = pickle.load(f)

    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score, \
    classification_report

    # Get predicted probabilities from your model
    y_scores = model.predict(X)  # Continuous scores with range ~150

    # Generate thresholds across your score range
    min_score = np.min(y_scores)
    max_score = np.max(y_scores)
    thresholds = np.linspace(min_score, max_score, 30)  # 100 evenly spaced thresholds

    # Calculate metrics for each threshold
    precisions = []
    recalls = []
    f1s = []
    accuracies = []

    for threshold in tqdm(thresholds):
        y_pred = (y_scores >= threshold).astype(int)
        precisions.append(precision_score(y_test, y_pred, sample_weight=sample_weights, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, sample_weight=sample_weights))
        f1s.append(f1_score(y_test, y_pred, sample_weight=sample_weights))
        accuracies.append(accuracy_score(y_test, y_pred, sample_weight=sample_weights))

    # Find optimal threshold (max F1-score)
    optimal_idx = np.argmax(f1s)
    optimal_threshold = thresholds[optimal_idx]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1s, label='F1-score')
    plt.plot(thresholds, accuracies, label='Accuracy')

    # Mark optimal threshold
    plt.axvline(optimal_threshold, color='red', linestyle='--',
                label=f'Optimal Threshold ({optimal_threshold:.2f})')

    plt.xlabel('Score Threshold')
    plt.ylabel('Metric Score')
    plt.title('Threshold Tuning for Continuous Scores (Weighted)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Optimal threshold: {optimal_threshold:.2f}")
    print(f"At this threshold:")
    print(f"- Precision: {precisions[optimal_idx]:.2f}")
    print(f"- Recall: {recalls[optimal_idx]:.2f}")
    print(f"- F1-score: {f1s[optimal_idx]:.2f}")
    print(f"- Accuracy: {accuracies[optimal_idx]:.2f}")

    # Generate predictions using the optimal threshold
    y_pred_optimal = (y_scores >= optimal_threshold).astype(int)

    # Full classification report
    print("\nClassification Report at Optimal Threshold:")
    print(classification_report(
        y_test, y_pred_optimal,
        sample_weight=sample_weights,
        target_names=['Class 0', 'Class 1']  # Replace with your class names
    ))

    plt.show()