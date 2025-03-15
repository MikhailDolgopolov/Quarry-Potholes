from typing import Literal

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

from pygam import LogisticGAM
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_sample_weight
from tqdm import tqdm
import pickle

from data_read import load_prepared

pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)

ws = 5
target = 'class'
big_df = load_prepared(f"data/class{ws}", sample_frac=0.5)

train_df, test_df = train_test_split(big_df, test_size=0.2)
X_train, y_train = train_df.drop(columns=[target]), train_df[target]
X_test, y_test = test_df.drop(columns=[target]), test_df[target]

sample_weights = compute_sample_weight('balanced', y_train)
lr=0.1
md = 4
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=lr,
    max_depth=md
)
model.fit(X_train, y_train, sample_weights)

y_pred = model.predict(X_test)

test_weights = compute_sample_weight('balanced', y_test)
mae = mean_absolute_error(y_test, y_pred, sample_weight=test_weights)
print(f"MAE: {mae:.2f}")
pic_path = f"models/GBR[lr{lr}][depth{md}]_[ws{ws}]-balanced{round(mae)}.pkl"

with open(pic_path, "wb") as f:
    pickle.dump(model, f)
