import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from data_read import load_prepared

model = XGBClassifier()
model.load_model('models/test.model')

big_df = load_prepared('data/prepared7')

X, y = big_df.drop(columns=['hole']), big_df['hole']

y_pred = model.predict(X)

print("Classification Report:")
print(classification_report(y, y_pred))