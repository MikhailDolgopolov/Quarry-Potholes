from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from pygam import LogisticGAM
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle

from data_read import load_prepared

pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)


def train_GB(folder_path: str, ws: int):
    # Load the prepared data using the folder_path and window size (ws)
    # For example, if folder_path='data/prepared' and ws=7, this loads 'data/prepared7'
    big_df = load_prepared(f'{folder_path}{ws}')

    dep_var = folder_path.split(r'/')[-1]
    # Split into training and testing sets
    train_df, test_df = train_test_split(big_df, test_size=0.2)

    # Use the provided dependent variable column for the target,
    # and drop that column from features.
    X_train = train_df.drop(columns=[dep_var])
    y_train = train_df[dep_var]
    X_test = test_df.drop(columns=[dep_var])
    y_test = test_df[dep_var]

    # Set up and train the model based on the target type
    if dep_var == 'hole':  # Binary classification case
        # Calculate the imbalance ratio (assuming class 0 is majority, 1 is minority)
        negative_count = (y_train == 0).sum()
        positive_count = (y_train == 1).sum()
        bias = 1.2  # Adjust this bias as needed
        scale_weight = negative_count / positive_count * bias
        print(
            f'Actual class proportion is {negative_count / positive_count:.2f}, biasing it towards potholes by {bias}: {scale_weight:.2f}')

        model = xgb.XGBClassifier(scale_pos_weight=scale_weight, max_depth=6)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif dep_var == 'class':

        # Encode factor labels as zero-based indices

        label_encoder = LabelEncoder()

        y_train_encoded = label_encoder.fit_transform(y_train)

        y_test_encoded = label_encoder.transform(y_test)  # Ensure test set uses same encoding

        n_classes = len(label_encoder.classes_)

        print(f'Multiclass classification with {n_classes} classes: {list(label_encoder.classes_)}')

        model = xgb.XGBClassifier(objective='multi:softprob', num_class=n_classes)

        # Train the model

        model.fit(X_train, y_train_encoded)

        # Predict using encoded labels

        y_pred_encoded = model.predict(X_test)

        # Convert predictions back to original labels

        y_pred = label_encoder.inverse_transform(y_pred_encoded)

        y_test = label_encoder.inverse_transform(y_test_encoded)


    else:

        raise ValueError("dep_var must be either 'hole' (binary) or 'class' (multiclass).")


    print("Classification Report:")
    print(classification_report(y_test, y_pred))


# train_GB(folder_path='data/hole', ws=10, dep_var='hole')

train_GB(folder_path='data/class', ws=10)
