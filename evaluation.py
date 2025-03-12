import pickle
import numpy as np
import pygam
from sklearn.metrics import classification_report
from data_read import load_prepared

ws = 7
big_df = load_prepared(f'data/prepared{ws}')
filename = f'models/gam{ws}.pkl'
with open(filename, 'rb') as f:
    model: pygam.LogisticGAM = pickle.load(f)

X_test, y_test = big_df.drop(columns=['hole']), big_df['hole']

y_pred = (model.predict_proba(X_test) > 0.35).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred))