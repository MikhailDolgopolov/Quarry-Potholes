import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from pygam import LogisticGAM
from tqdm import tqdm
import pickle

from data_read import load_prepared

pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)

ws = 10
big_df = load_prepared(f'data/prepared{ws}')
train_df, test_df = train_test_split(big_df, test_size=0.2)

X_train, y_train = train_df.drop(columns=['hole']), train_df['hole']
X_test, y_test = test_df.drop(columns=['hole']), test_df['hole']

negative_count = (y_train == 0).sum()  # Assuming class 0 is the majority
positive_count = (y_train == 1).sum()  # Assuming class 1 is the minority
bias = 1.2
scale_weight = negative_count / positive_count * bias
print(f'Actual class proportion is {negative_count / positive_count:.2f}, biasing it towards potholes by {bias}: {scale_weight:.2f}')

model = xgb.XGBClassifier(scale_pos_weight=scale_weight, max_depth=6)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
