import json
import pprint
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from tqdm import trange

from exploration.data_read import load_engineered_data
from helpers import train_split_by_column, split_off_target_cols




if __name__ == '__main__':
    # select_by_mannwhitney(num_cols=12, ws=7, data_path='data/engineered/ws30_peaks/rolled7')
    select_by_predictors(12, 7, 'data/engineered/ws30_peaks/rolled7')
