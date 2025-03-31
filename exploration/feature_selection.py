import json
import random
from collections import Counter
from pprint import pprint


def aggregate_to_three_sets(ws_data):
    """Aggregate features into three maximally different sets for a given window size."""
    # Flatten all features into a single list per ws
    all_features = []
    for method in ws_data.values():
        for sample in method:
            all_features.extend(sample)

    # Count feature frequencies
    feature_counts = Counter(all_features)
    total_instances = sum(len(samples) for samples in ws_data.values())  # Total number of feature lists

    # Sort features by frequency (most common to least common)
    sorted_features = sorted(feature_counts.items(), key=lambda x: (-x[1], x[0]))

    # Split into three sets based on frequency
    common = [feat for feat, count in sorted_features if count >= total_instances * 0.7]  # Appears in 70%+ of lists
    moderate = [feat for feat, count in sorted_features if total_instances * 0.3 <= count < total_instances * 0.7]
    diff = 10 - len(common) - len(moderate)
    if diff < 0:
        return {'v1': [*common, *random.choices(moderate, k=-diff)],
                'v2': [*common, *random.choices(moderate, k=-diff)]
                }
    else:
        return {
                'v1': [*common, *moderate]
                }


if __name__ == "__main__":
    with open("exploration/feature_selection_sets.json", "r") as f:
        all_sets=json.load(f)

    # Aggregate data and save to JSON
    aggregated_results = {ws: aggregate_to_three_sets(data) for ws, data in all_sets.items()}
    with open("exploration/features_selected.json", "w") as f:
        json.dump(aggregated_results, f, indent=4)